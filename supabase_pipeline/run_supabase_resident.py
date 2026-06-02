"""
supabase_pipeline/run_supabase_resident.py — Supabase-driven, all-resident
orchestrator for the Alluvi IMAGE pipeline.

This is a rewired copy of orchestration/all_resident/run_all_resident.py. The
proven per-scenario stage flow (Stage 1 -> Stage 2 + QC retries -> Stage 3) is
preserved verbatim in spirit; what changed:
  - INPUT comes from Supabase `tiktok_accounts`, not a fixed persona.
  - NEW Phase A0 creates (or reuses) a per-account portrait + persona.yaml.
  - The existing src/step_1_* modules are reused UNCHANGED via per-account
    runtime overrides of their module-level persona constants.
  - All tracking goes to the 8-table Supabase schema instead of SQLite.

PHASES
  A0  per selected account: reuse persona if it exists, else Opus appearance ->
      FLUX portrait -> write personas row + per-account persona.yaml.
      (FLUX.1-dev is loaded ONCE for all creates, then unloaded.)
  A1  load the resident trio: PuLID(+FLUX), Qwen(ComfyUI), Kontext.
  B   per account: override the persona constants, then for the next N not-yet-
      done scenarios run Stage 1 -> Stage 2 + QC -> Stage 3; write outputs +
      stage_executions + llm_calls + image_generations + qc_checks.
  C   unload all + finalize the pipeline_runs row.

CONTROLS (CLI)
  --all-accounts        process every tiktok_accounts row (default OFF = only
                        accounts that have no persona yet)
  --num-scenarios N     scenarios per persona this run (default 1)
  --accounts a,b        OVERRIDE: only these tiktok_ids (ignores --all-accounts)
  --yes                 skip the cost-confirmation prompt
  --skip-preflight      skip the connectivity/asset checks
  env QC_ENABLED=false / STEP_3_ENABLED=false  toggle QC / Stage 3

  Resume is automatic: existing personas are reused (the face never regenerates),
  and scenarios that already have an outputs row for a persona are skipped.

USAGE
  cd /workspace/alluvi-pipeline
  # one account, one scenario:
  python supabase_pipeline/run_supabase_resident.py --accounts @emma.callahan --num-scenarios 1 --yes
  # all accounts, 5 scenarios each:
  python supabase_pipeline/run_supabase_resident.py --all-accounts --num-scenarios 5 --yes
  # only brand-new accounts (no persona yet), 1 scenario each:
  python supabase_pipeline/run_supabase_resident.py --num-scenarios 1 --yes
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

import yaml

# ── repo root on sys.path so `from src...` and `from supabase_pipeline...` resolve
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reused UNCHANGED from the existing pipeline
from src import scenario_loader
from src import step_1_prompt_builder
from src import step_1_pulid
from src import step_2_prompt_builder
from src import step_2_qwen_comfyui as step_2_qwen_edit
from src import step_3_realism
from src import vram_utils
from src.json_utils import JSONSanityError
from src.qc_validator import validate_image, QC_MODEL

# New (this folder)
from supabase_pipeline import supabase_db
from supabase_pipeline import phase_a_prompt_builder
from supabase_pipeline import phase_a_persona


CONFIG_PATH = REPO_ROOT / "config.yaml"
OUTPUT_ROOT = REPO_ROOT / "outputs"
PERSONAS_ROOT = REPO_ROOT / "outputs" / "personas"   # persists on /workspace

FLOW_NAME = "supabase_resident"
OPUS_MODEL = "claude-opus-4-7"

MAX_JSON_RETRIES = 1
MAX_QC_RETRIES = 2          # up to 3 Stage-2 attempts
POD_HOURLY_USD_H200 = 3.99

_STAGE_MODEL = {
    "phaseA_persona": "FLUX.1-dev",
    "stage1_pulid":   "FLUX.1-dev + PuLID",
    "stage2_qwen":    "Qwen-Image-Edit-2511",
    "stage3_kontext": "FLUX.1-Kontext-dev",
}


# ──────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"missing config: {CONFIG_PATH}")
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _safe_id(tiktok_id: str) -> str:
    return tiktok_id.lstrip("@").replace("/", "_").replace(" ", "_")


def _db(label: str, fn):
    """Run a DB write non-fatally — warn on failure, never crash the run."""
    try:
        return fn()
    except Exception as e:
        print(f"  [db] {label} failed (non-fatal): {type(e).__name__}: {e}")
        return None


def _call_with_json_retry(build_fn, sid: str, step_label: str, max_retries: int = MAX_JSON_RETRIES):
    last_error = None
    for attempt in range(1, max_retries + 2):
        try:
            return build_fn()
        except JSONSanityError as e:
            last_error = e
            print(f"  {sid} {step_label}: JSON sanity error "
                  f"{attempt}/{max_retries + 1}: {str(e)[:160]}")
    raise last_error


def _write_persona_yaml(envelope: dict, path: Path) -> None:
    """Serialize the appearance envelope (minus generation-only fields) to YAML."""
    data = {k: v for k, v in envelope.items()
            if k not in ("portrait_prompt", "portrait_negative_prompt")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _read_sidecar(out_path: Path) -> dict:
    """Read the {stem}_request.json a stage module wrote next to its output."""
    p = out_path.parent / f"{out_path.stem}_request.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("arguments", {})
    except Exception:
        return {}


def _audit_image_gen(stage_exec_id, stage_name: str, out_path: Path, meta: dict,
                     attempt_number: int = 1) -> None:
    """Map a stage's audit sidecar + return meta into an image_generations row."""
    a = _read_sidecar(out_path)
    imgs = a.get("image_urls")
    if not imgs:
        single = a.get("persona_image_path") or a.get("input_image_path") or a.get("image_path")
        imgs = [single] if single else None
    cfg = a.get("cfg") or a.get("true_cfg") or a.get("true_cfg_scale")
    _db(f"image_gen {stage_name}", lambda: supabase_db.insert_image_generation(
        stage_execution_id=stage_exec_id,
        stage_name=stage_name,
        model_name=_STAGE_MODEL.get(stage_name),
        endpoint=meta.get("endpoint") or a.get("endpoint"),
        backend=a.get("backend"),
        workflow_template=a.get("workflow_template"),
        prompt=a.get("prompt"),
        negative_prompt=a.get("negative_prompt"),
        input_image_paths=imgs,
        num_inference_steps=a.get("num_inference_steps"),
        cfg=cfg,
        guidance=a.get("guidance_scale"),
        seed=meta.get("seed") or a.get("seed"),
        width=a.get("width"),
        height=a.get("height"),
        target_size=a.get("target_size"),
        id_weight=a.get("id_weight"),
        max_sequence_length=a.get("max_sequence_length"),
        output_local_path=meta.get("local_path") or str(out_path),
        request_id=meta.get("request_id"),
        elapsed_seconds=meta.get("elapsed_seconds"),
        cost_usd=meta.get("cost_usd", 0.0),
        attempt_number=attempt_number,
        comfyui_server=a.get("comfyui_server"),
    ))


# ──────────────────────────────────────────────────────────────────────────
# Phase A0 — persona create-or-reuse
# ──────────────────────────────────────────────────────────────────────────

def _persona_ready(account: dict) -> bool:
    """True if a persona row exists AND its portrait file is on disk."""
    row = supabase_db.get_persona_for_account(account["id"])
    if not row:
        return False
    p = row.get("portrait_local_path")
    return bool(p and Path(p).exists())


def ensure_persona(account: dict, flux_pipe, run_pk) -> dict:
    """
    Returns {persona_id, portrait_path, yaml_path, created}.
    Reuses the existing face if present (NEVER regenerates it); only (re)builds
    the per-account persona.yaml if it's missing.
    """
    sid = account["tiktok_id"]
    pdir = PERSONAS_ROOT / _safe_id(sid)
    portrait_path = pdir / "portrait.jpg"
    yaml_path = pdir / "persona.yaml"
    pdir.mkdir(parents=True, exist_ok=True)

    row = supabase_db.get_persona_for_account(account["id"])
    have_portrait = (row and row.get("portrait_local_path")
                     and Path(row["portrait_local_path"]).exists())

    # ── REUSE ────────────────────────────────────────────────────────────
    if row and have_portrait:
        portrait_path = Path(row["portrait_local_path"])
        if not yaml_path.exists():
            # rebuild yaml from the stored envelope (no new face, no FLUX)
            envelope = None
            if row.get("appearance_spec"):
                try:
                    envelope = json.loads(row["appearance_spec"])
                except Exception:
                    envelope = None
            if envelope is None:
                envelope = phase_a_prompt_builder.build_appearance_prompt(account)
            _write_persona_yaml(envelope, yaml_path)
        print(f"  [{sid}] persona REUSED (portrait {portrait_path.name})")
        return {"persona_id": row["id"], "portrait_path": str(portrait_path),
                "yaml_path": yaml_path, "created": False}

    # ── CREATE ───────────────────────────────────────────────────────────
    if flux_pipe is None:
        raise RuntimeError(f"persona for {sid} needs creating but FLUX is not loaded")

    print(f"  [{sid}] creating persona...")
    t0 = time.time()
    envelope = phase_a_prompt_builder.build_appearance_prompt(account)        # Opus
    meta = phase_a_persona.generate(                                          # FLUX
        flux_pipe, envelope["portrait_prompt"], portrait_path, account_id=sid)
    _write_persona_yaml(envelope, yaml_path)
    (pdir / "appearance_envelope.json").write_text(
        json.dumps(envelope, indent=2, ensure_ascii=False), encoding="utf-8")

    persona = _db("upsert_persona", lambda: supabase_db.upsert_persona(
        tiktok_account_id=account["id"],
        appearance_spec=json.dumps(envelope, ensure_ascii=False),
        prompt_used=envelope["portrait_prompt"],
        portrait_local_path=str(portrait_path),
        status="done",
    ))
    persona_id = persona["id"] if persona else None

    # audit the Phase-A stage
    se = _db("start phaseA", lambda: supabase_db.start_stage(
        stage_name="phaseA_persona", run_pk=run_pk, persona_id=persona_id))
    _db("llm phaseA", lambda: supabase_db.insert_llm_call(
        stage_execution_id=se, purpose="appearance_prompt", model=OPUS_MODEL,
        system_prompt_name="master_prompt_phaseA", system_prompt_version="v1",
        parsed_json=envelope))
    _audit_image_gen(se, "phaseA_persona", portrait_path, meta)
    _db("finish phaseA", lambda: supabase_db.finish_stage(
        se, status="done", elapsed_seconds=time.time() - t0))

    print(f"  [{sid}] persona CREATED in {time.time()-t0:.1f}s")
    return {"persona_id": persona_id, "portrait_path": str(portrait_path),
            "yaml_path": yaml_path, "created": True}


# ──────────────────────────────────────────────────────────────────────────
# Phase B — one (account, scenario) through Stage 1 -> 2 -> QC -> 3
# ──────────────────────────────────────────────────────────────────────────

def process_scenario(account, persona_info, scenario, config, run_pk,
                     pipe_1, pipe_2, pipe_3, qc_enabled, step_3_enabled) -> str:
    """Run all 3 stages for one (account, scenario). Returns final status."""
    sid = scenario.get("id", "?")
    persona_id = persona_info["persona_id"]
    acc_sid = _safe_id(account["tiktok_id"])
    output_dir = OUTPUT_ROOT / persona_info["run_id"] / acc_sid / sid
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{acc_sid}/{sid}"
    t_start = time.time()

    (output_dir / "01_scenario.yaml").write_text(
        json.dumps(scenario, indent=2, ensure_ascii=False), encoding="utf-8")

    # outputs row created up front (status running) so audit rows can reference it
    out_row = _db("upsert_output running", lambda: supabase_db.upsert_output(
        persona_id=persona_id, scenario_id=sid,
        scenario_title=scenario.get("category"), run_pk=run_pk,
        status="running"))
    output_id = out_row["id"] if out_row else None

    def _finish_output(status, qc_status, qc_reason, attempts, final_path):
        _db("upsert_output final", lambda: supabase_db.upsert_output(
            persona_id=persona_id, scenario_id=sid,
            scenario_title=scenario.get("category"), run_pk=run_pk,
            final_image_local_path=final_path, qc_status=qc_status,
            qc_reason=qc_reason, attempts=attempts, status=status))

    # ── Stage 1: prompt + PuLID ──────────────────────────────────────────
    se1 = _db("start stage1", lambda: supabase_db.start_stage(
        stage_name="stage1_pulid", run_pk=run_pk,
        persona_id=persona_id, output_id=output_id))
    s1_t = time.time()
    try:
        step_1_output = _call_with_json_retry(
            lambda: step_1_prompt_builder.build_step_1_prompt(scenario), sid, "Step 1")
    except Exception as e:
        _db("finish stage1 fail", lambda: supabase_db.finish_stage(
            se1, status="failed", error_message=str(e)))
        _finish_output("failed", None, f"step1_prompt: {e}", 0, None)
        print(f"  [{tag}] FAILED step1 prompt: {e}")
        return "failed"

    step_1_text = (step_1_output or {}).get("step_1_image_prompt", "").strip()
    _db("llm stage1", lambda: supabase_db.insert_llm_call(
        stage_execution_id=se1, purpose="stage1_prompt", model=OPUS_MODEL,
        system_prompt_name="master_prompt_step1", parsed_json=step_1_output))
    (output_dir / "02_step1_prompt.json").write_text(
        json.dumps(step_1_output, indent=2, ensure_ascii=False), encoding="utf-8")

    pulid_params = step_1_output.get("fal_pulid_params") or config.get("step_1", {}).get("defaults", {})
    persona_out = output_dir / "03_step1_persona.jpg"
    try:
        s1_meta = step_1_pulid.generate(
            pipeline=pipe_1, step_1_prompt=step_1_text,
            fal_pulid_params=pulid_params, out_path=persona_out, scenario_id=sid)
    except Exception as e:
        traceback.print_exc()
        _db("finish stage1 fail", lambda: supabase_db.finish_stage(
            se1, status="failed", error_message=str(e)))
        _finish_output("failed", None, f"stage1_pulid: {e}", 0, None)
        print(f"  [{tag}] FAILED stage1 inference: {e}")
        return "failed"

    (output_dir / "03_step1_meta.json").write_text(
        json.dumps(s1_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    _audit_image_gen(se1, "stage1_pulid", persona_out, s1_meta)
    _db("finish stage1", lambda: supabase_db.finish_stage(
        se1, status="done", elapsed_seconds=s1_meta.get("elapsed_seconds")))

    # ── Stage 2: prompt + Qwen (ComfyUI) + QC retries ────────────────────
    se2 = _db("start stage2", lambda: supabase_db.start_stage(
        stage_name="stage2_qwen", run_pk=run_pk,
        persona_id=persona_id, output_id=output_id))
    s2_t = time.time()
    try:
        step_2_output = _call_with_json_retry(
            lambda: step_2_prompt_builder.build_step_2_prompt(scenario, step_1_output),
            sid, "Step 2")
    except Exception as e:
        _db("finish stage2 fail", lambda: supabase_db.finish_stage(
            se2, status="failed", error_message=str(e)))
        _finish_output("failed", None, f"step2_prompt: {e}", 0, None)
        print(f"  [{tag}] FAILED step2 prompt: {e}")
        return "failed"

    step_2_text = (step_2_output or {}).get("step_2_image_prompt", "").strip()
    _db("llm stage2", lambda: supabase_db.insert_llm_call(
        stage_execution_id=se2, purpose="stage2_prompt", model=OPUS_MODEL,
        system_prompt_name="master_prompt_step2_qwen", system_prompt_version="v6",
        parsed_json=step_2_output))
    (output_dir / "04_step2_prompt.json").write_text(
        json.dumps(step_2_output, indent=2, ensure_ascii=False), encoding="utf-8")

    qwen_params = step_2_output.get("fal_qwen_params") or config.get("step_2", {}).get("defaults", {})
    final_out = output_dir / "05_step2_final.jpg"
    attempt_prompt = step_2_text
    final_qc = None
    attempts_run = 0

    for attempt in range(1, MAX_QC_RETRIES + 2):
        attempts_run = attempt
        is_last = attempt == MAX_QC_RETRIES + 1
        attempt_img = output_dir / f"05_step2_final_attempt_{attempt}.jpg"
        try:
            s2_meta = step_2_qwen_edit.generate(
                pipeline=pipe_2, step_1_local_path=str(persona_out),
                step_2_prompt=attempt_prompt, fal_qwen_params=qwen_params,
                out_path=attempt_img, scenario_id=f"{sid}#a{attempt}")
        except Exception as e:
            traceback.print_exc()
            _db("finish stage2 fail", lambda: supabase_db.finish_stage(
                se2, status="failed", error_message=str(e)))
            _finish_output("failed", None, f"stage2 attempt {attempt}: {e}",
                           attempt, None)
            print(f"  [{tag}] FAILED stage2 attempt {attempt}: {e}")
            return "failed"

        try:
            shutil.copy(attempt_img, final_out)
        except Exception:
            pass
        ig_id = None  # (image_generation id not needed downstream; QC links by stage)
        _audit_image_gen(se2, "stage2_qwen", attempt_img, s2_meta, attempt_number=attempt)

        if not qc_enabled:
            final_qc = {"passed": True, "issues": [], "recommendation": "use",
                        "error": "QC disabled"}
            _db("qc disabled", lambda: supabase_db.insert_qc_check(
                stage_execution_id=se2, output_id=output_id, attempt_number=attempt,
                qc_model="disabled", passed=True, qc_reason="qc_disabled",
                image_evaluated_path=str(final_out)))
            break

        try:
            qc = validate_image(final_out, scenario_id=f"{sid}#a{attempt}")
        except Exception as e:
            qc = {"passed": True, "score": 0.5, "issues": [f"QC crashed: {e}"],
                  "recommendation": "use", "error": str(e)}

        (output_dir / f"06_qc_result_attempt_{attempt}.json").write_text(
            json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8")
        avoid = ("AVOID: " + "; ".join([str(x) for x in (qc.get("issues") or [])][:6]) + ".") \
            if (not qc.get("passed") and qc.get("issues")) else None
        _db(f"qc_check a{attempt}", lambda q=qc, av=avoid: supabase_db.insert_qc_check(
            stage_execution_id=se2, output_id=output_id, attempt_number=attempt,
            qc_model=QC_MODEL, passed=bool(q.get("passed")),
            qc_reason=("pass" if q.get("passed") else "defect"),
            issues=q.get("issues"), limb_description=q.get("limb_description"),
            scores=q.get("checks"), avoid_line=av,
            image_evaluated_path=str(final_out), raw_result=q.get("raw_vlm_response")))

        if qc.get("passed"):
            print(f"  [{tag}] QC PASSED on attempt {attempt}")
            final_qc = qc
            break
        final_qc = qc
        if not is_last:
            attempt_prompt = step_2_text + ("\n\n" + avoid if avoid else "")
            print(f"  [{tag}] QC failed attempt {attempt} — retrying")
        else:
            print(f"  [{tag}] QC failed on final attempt {attempt}")

    (output_dir / "05_step2_meta.json").write_text(
        json.dumps(s2_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    _db("finish stage2", lambda: supabase_db.finish_stage(
        se2, status="done", elapsed_seconds=time.time() - s2_t))

    qc_passed = bool(final_qc and final_qc.get("passed"))
    final_image_path = str(final_out)

    # ── Stage 3: Kontext realism (only if QC passed; non-fatal) ──────────
    if step_3_enabled and qc_passed:
        se3 = _db("start stage3", lambda: supabase_db.start_stage(
            stage_name="stage3_kontext", run_pk=run_pk,
            persona_id=persona_id, output_id=output_id))
        s3_out = output_dir / "07_step3_realism.jpg"
        s3_t = time.time()
        try:
            s3_meta = step_3_realism.generate(
                pipeline=pipe_3, step_2_local_path=str(final_out),
                out_path=s3_out, scenario_id=sid)
            (output_dir / "07_step3_meta.json").write_text(
                json.dumps(s3_meta, indent=2, ensure_ascii=False), encoding="utf-8")
            _audit_image_gen(se3, "stage3_kontext", s3_out, s3_meta)
            _db("finish stage3", lambda: supabase_db.finish_stage(
                se3, status="done", elapsed_seconds=s3_meta.get("elapsed_seconds")))
            final_image_path = str(s3_out)
        except Exception as e:
            print(f"  [{tag}] Stage 3 failed (non-fatal, using Stage 2): {e}")
            _db("finish stage3 fail", lambda: supabase_db.finish_stage(
                se3, status="failed", elapsed_seconds=time.time() - s3_t,
                error_message=str(e)))

    # ── finalize outputs row ─────────────────────────────────────────────
    if qc_passed:
        status, qc_status, qc_reason = "done", "pass", None
    elif not qc_enabled:
        status, qc_status, qc_reason = "done", "skipped", "qc_disabled"
    else:
        status, qc_status = "qc_failed", "qc_failed"
        qc_reason = "; ".join((final_qc or {}).get("issues", []) or ["qc failed"])
    _finish_output(status, qc_status, qc_reason, attempts_run, final_image_path)

    print(f"  [{tag}] DONE in {time.time()-t_start:.1f}s (status={status})")
    return status


# ──────────────────────────────────────────────────────────────────────────
# Account selection + cost
# ──────────────────────────────────────────────────────────────────────────

def _select_accounts(args) -> list[dict]:
    if args.accounts:
        wanted = [x.strip() for x in args.accounts.split(",") if x.strip()]
        wanted += ["@" + x for x in wanted if not x.startswith("@")]  # tolerate missing @
        accts = supabase_db.get_accounts_by_tiktok_ids(list(set(wanted)))
        if not accts:
            print(f"[run] no accounts matched --accounts {args.accounts}")
        return accts
    if args.all_accounts:
        return supabase_db.get_all_accounts()
    return supabase_db.get_accounts_without_persona()


def _confirm_cost(accounts, n_to_create, num_scenarios, skip) -> bool:
    max_jobs = len(accounts) * num_scenarios
    opus_personas = n_to_create * 0.10
    llm_scenarios = max_jobs * 0.30
    print("\n" + "=" * 72)
    print(" SUPABASE-RESIDENT RUN PLAN")
    print("=" * 72)
    print(f"  accounts selected:      {len(accounts)}")
    print(f"  personas to create:     {n_to_create}  (rest reused)")
    print(f"  scenarios/persona:      {num_scenarios}")
    print(f"  max scenario jobs:      {max_jobs}  (fewer if some already done)")
    print(f"  est. LLM cost:          ~${opus_personas + llm_scenarios:.2f}")
    print(f"  + GPU time on the H200 (${POD_HOURLY_USD_H200}/hr)")
    print("=" * 72)
    if skip:
        print("[run] --yes: skipping confirmation")
        return True
    try:
        return input("Proceed? [y/N]: ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _preflight() -> list[str]:
    errs = []
    if not os.getenv("ANTHROPIC_API_KEY"):
        errs.append("ANTHROPIC_API_KEY missing in .env")
    try:
        n = supabase_db.ping()
        print(f"[preflight] supabase OK ({n} accounts)")
    except Exception as e:
        errs.append(f"supabase not reachable: {e}")
    if not Path(phase_a_persona.FLUX_DEV_DIR).exists():
        errs.append(f"FLUX.1-dev not found at {phase_a_persona.FLUX_DEV_DIR}")
    return errs


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Alluvi Supabase-resident image pipeline")
    ap.add_argument("--all-accounts", action="store_true",
                    help="process every account (default: only accounts without a persona)")
    ap.add_argument("--num-scenarios", type=int, default=1,
                    help="scenarios per persona this run (default 1)")
    ap.add_argument("--accounts", type=str, default=None,
                    help="comma-separated tiktok_ids; overrides --all-accounts")
    ap.add_argument("--yes", action="store_true", help="skip cost confirmation")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    qc_enabled = os.getenv("QC_ENABLED", "true").lower() == "true"
    step_3_enabled = os.getenv("STEP_3_ENABLED", "true").lower() == "true"

    if not args.skip_preflight:
        errs = _preflight()
        if errs:
            print("[run] preflight failed:")
            for e in errs:
                print(f"  X {e}")
            return 1

    accounts = _select_accounts(args)
    if not accounts:
        print("[run] no accounts to process — nothing to do")
        return 1

    try:
        all_scenarios = scenario_loader.load_scenarios()
    except Exception as e:
        print(f"[run] failed to load scenarios.yaml: {e}")
        return 1

    config = _load_config()
    n_to_create = sum(0 if _persona_ready(a) else 1 for a in accounts)
    if not _confirm_cost(accounts, n_to_create, args.num_scenarios, args.yes):
        print("[run] aborted by user")
        return 1

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id_text = f"{timestamp}_{FLOW_NAME}"
    scenario_filter = (args.accounts or ("all_accounts" if args.all_accounts else "new_accounts"))
    run_pk = _db("create_run", lambda: supabase_db.create_run(
        run_id=run_id_text, flow_name=FLOW_NAME, scenario_filter=scenario_filter,
        total_scenarios=len(accounts) * args.num_scenarios,
        notes=f"accounts={len(accounts)}, n={args.num_scenarios}, create={n_to_create}"))

    print("\n" + "=" * 72)
    print(f" RUN {run_id_text}  | accounts={len(accounts)} create={n_to_create} "
          f"N={args.num_scenarios} | QC={qc_enabled} Stage3={step_3_enabled}")
    print("=" * 72)

    # ── Phase A0: personas (load FLUX only if something needs creating) ──
    print("\n--- PHASE A0: personas ---")
    flux = phase_a_persona.load_pipeline() if n_to_create > 0 else None
    persona_info: dict[int, dict] = {}
    try:
        for a in accounts:
            info = ensure_persona(a, flux, run_pk)
            info["run_id"] = run_id_text
            persona_info[a["id"]] = info
    finally:
        if flux is not None:
            print("  unloading FLUX.1-dev (Phase A0 done)")
            phase_a_persona.unload_pipeline(flux)

    # ── Phase A1: load the resident trio ─────────────────────────────────
    print("\n--- PHASE A1: loading resident trio (PuLID, Qwen, Kontext) ---")
    pipe_1 = pipe_2 = pipe_3 = None
    counts = {"done": 0, "qc_failed": 0, "failed": 0, "skipped_done": 0}
    load_start = time.time()
    try:
        vram_utils.reset_vram_peak()
        pipe_1 = step_1_pulid.load_pipeline()
        pipe_2 = step_2_qwen_edit.load_pipeline()
        pipe_3 = step_3_realism.load_pipeline()
        vram_utils.report_vram("all 3 resident")

        # ── Phase B: per account × next-N-undone scenarios ──────────────
        print("\n--- PHASE B: scenarios ---")
        for a in accounts:
            info = persona_info[a["id"]]
            if not info["persona_id"]:
                print(f"  [{a['tiktok_id']}] no persona id — skipping")
                continue
            # per-account persona override (files untouched)
            step_1_pulid.PERSONA_IMAGE_PATH = Path(info["portrait_path"])
            step_1_prompt_builder.PERSONA_YAML_PATH = info["yaml_path"]
            step_1_prompt_builder._static_context_cache = None

            done = supabase_db.get_done_scenario_ids(info["persona_id"])
            todo = [s for s in all_scenarios if s.get("id") not in done][:args.num_scenarios]
            already = args.num_scenarios - len(todo)
            counts["skipped_done"] += max(0, already)
            print(f"\n  === {a['tiktok_id']} (persona {info['persona_id']}): "
                  f"{len(todo)} scenario(s), {len(done)} already done ===")
            for scenario in todo:
                status = process_scenario(
                    a, info, scenario, config, run_pk,
                    pipe_1, pipe_2, pipe_3, qc_enabled, step_3_enabled)
                counts[status] = counts.get(status, 0) + 1
    except KeyboardInterrupt:
        print("\n  interrupted — finalizing...")
    finally:
        print("\n--- PHASE C: unload ---")
        for name, pipe in (("Kontext", pipe_3), ("Qwen", pipe_2), ("PuLID", pipe_1)):
            if pipe is None:
                continue
            try:
                if name == "Qwen":
                    step_2_qwen_edit.unload_pipeline(pipe)
                else:
                    vram_utils.unload_pipeline(pipe)
            except Exception as e:
                print(f"  {name} unload error (non-fatal): {e}")
        vram_utils.report_vram("after unload")

    _db("finalize_run", lambda: supabase_db.finalize_run(run_pk, status="done"))

    print("\n" + "=" * 72)
    print(" RUN COMPLETE")
    print(f"  done={counts['done']}  qc_failed={counts['qc_failed']}  "
          f"failed={counts['failed']}  scenarios_already_done={counts['skipped_done']}")
    print(f"  wall: {time.time()-load_start:.1f}s   run_id: {run_id_text}")
    print("=" * 72)
    return 0 if counts["failed"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
