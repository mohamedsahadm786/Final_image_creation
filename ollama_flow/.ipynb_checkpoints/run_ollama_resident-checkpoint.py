"""
ollama_flow/run_ollama_resident.py — H200 all-resident runner for the Ollama flow.

ARCHITECTURE — all 3 pipelines stay resident in VRAM for the entire batch.

  Phase A (one-time, ~3-5 min):  load all 3 pipelines into VRAM
  Phase B (per scenario, ~2-3 min each):
                                 Ollama Step 1 prompt → Stage 1 inference
                                 → Ollama Step 2 prompt → Stage 2 inference
                                 → Stage 3 inference
                                 → write per-scenario chain.html
  Phase C (one-time):            unload all + write overview.html + manifest

  Total model loads:             3 (regardless of N scenarios)
  Per-scenario latency:          ~2-3 min (PURE inference — no load overhead,
                                 no LLM cost since Ollama is local)
  vs run_ollama.py (load/unload): shaves ~60s/scenario × N

  REQUIRES H200 (141 GB VRAM). Will OOM on Blackwell 96 GB or H100 80 GB —
  use run_ollama.py on those.

PRE-LOAD VRAM (steady state after Phase A):
  pipe_1 (FLUX-dev + PuLID stack):       ~28 GB
  pipe_2 (Qwen-Image-Edit-2511):         ~40-48 GB
  pipe_3 (FLUX.1-Kontext-dev):           ~24 GB
  total resident:                        ~92-100 GB
  peak during Stage 2 inference:         ~110-120 GB
  H200 SXM available:                    141 GB

  Note: Ollama's qwen2.5:7b ALSO runs on the GPU (~5 GB when loaded).
  Total peak: ~115-125 GB. Still comfortable under 141 GB.

NO QC in Ollama flow (no Sonnet validator). Stage 3 runs unconditionally
after every successful Stage 2.

CLI:
    python run_ollama_resident.py --scenario travel_hotel_morning_29
    python run_ollama_resident.py --pilot
    python run_ollama_resident.py --only ID1,ID2
    python run_ollama_resident.py
"""

import argparse
import json
import os
import sys
import time
import traceback
import uuid
import yaml
from datetime import datetime
from pathlib import Path

# Path setup — ollama_flow is at <repo>/ollama_flow/
OLLAMA_FLOW_ROOT = Path(__file__).resolve().parent
PARENT_REPO_ROOT = OLLAMA_FLOW_ROOT.parent

sys.path.insert(0, str(OLLAMA_FLOW_ROOT))
sys.path.insert(0, str(PARENT_REPO_ROOT))

# Redirect DB to alluvi_ollama.db BEFORE any DB call
from src import db as _db
_db.DB_PATH = OLLAMA_FLOW_ROOT / "data" / "alluvi_ollama.db"

# Parent's infrastructure
from src import scenario_loader
from src import step_1_pulid
from src import step_2_qwen_edit
from src import step_3_realism
from src import trace_html
from src import overview_html
from src import vram_utils
from src.json_utils import JSONSanityError

# Ollama prompt builders
from ollama_src import step_1_prompt_builder_ollama
from ollama_src import step_2_prompt_builder_ollama


CONFIG_PATH = OLLAMA_FLOW_ROOT / "config.yaml"
OUTPUT_ROOT = OLLAMA_FLOW_ROOT / "outputs"

PLAN_LABEL = "pulid_qwen_kontext_local_all_resident_ollama"

PILOT_COUNT = 5
MAX_JSON_RETRIES = 2  # Ollama 7B occasionally malforms JSON; retries are free
POD_HOURLY_USD_H200 = 3.99  # H200 SXM typical on-demand rate

DEFAULT_WALL_TIME_PER_SCENARIO_S = 160  # ~2.5 min average


# ──────────────────────────────────────────────────────────────────────────
# Config + helpers
# ──────────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"missing config: {CONFIG_PATH}")
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _call_with_json_retry(build_fn, scenario_id: str, step_label: str, max_retries: int = MAX_JSON_RETRIES):
    last_error = None
    for attempt in range(1, max_retries + 2):
        try:
            return build_fn()
        except JSONSanityError as e:
            last_error = e
            if attempt <= max_retries:
                print(f"  {scenario_id} {step_label}: JSON sanity error attempt "
                      f"{attempt}/{max_retries + 1}: {str(e)[:200]}")
            else:
                print(f"  {scenario_id} {step_label}: all {max_retries + 1} attempts failed")
    raise last_error


def _filter_scenarios(all_scenarios, only=None, exclude=None, pilot=False, single=None):
    if single:
        filtered = [s for s in all_scenarios if s.get("id") == single]
        if not filtered:
            print(f"[ollama_resident] scenario id not found: {single}")
        return filtered
    if only:
        only_set = {x.strip() for x in only.split(",") if x.strip()}
        filtered = [s for s in all_scenarios if s.get("id") in only_set]
        missing = only_set - {s.get("id") for s in filtered}
        if missing:
            print(f"[ollama_resident] WARNING: --only IDs not found: {sorted(missing)}")
    else:
        filtered = list(all_scenarios)
    if exclude:
        exclude_set = {x.strip() for x in exclude.split(",") if x.strip()}
        filtered = [s for s in filtered if s.get("id") not in exclude_set]
    if pilot:
        filtered = filtered[:PILOT_COUNT]
    return filtered


def _confirm_cost(n_scenarios, skip=False) -> bool:
    wall_seconds = 5 * 60 + n_scenarios * DEFAULT_WALL_TIME_PER_SCENARIO_S
    wall_hours = wall_seconds / 3600
    gpu_cost_total = wall_hours * POD_HOURLY_USD_H200

    print("")
    print("=" * 72)
    print(" BATCH PLAN (Ollama all-resident — H200 SXM)")
    print("=" * 72)
    print(f"  scenarios:           {n_scenarios}")
    print(f"  one-time load:       ~5 min")
    print(f"  per-scenario:        ~{DEFAULT_WALL_TIME_PER_SCENARIO_S}s (pure inference, no load overhead)")
    print(f"  LLM cost:            $0.00  (Ollama runs locally)")
    print(f"  Wall time (est):     ~{wall_seconds/60:.0f} min total")
    print(f"  GPU cost (est):      ~${gpu_cost_total:.2f}  ({wall_hours:.2f}h × ${POD_HOURLY_USD_H200}/hr)")
    print("=" * 72)
    print("")

    if skip:
        print("[ollama_resident] --yes flag: skipping confirmation")
        return True
    try:
        response = input("Proceed? [y/N]: ").strip().lower()
    except EOFError:
        response = ""
    return response in ("y", "yes")


def _init_record(scenario, output_dir, gen_id, run_id, model_label) -> dict:
    return {
        "scenario": scenario,
        "model_label": model_label,
        "output_dir": str(output_dir),
        "output_dir_path": output_dir,
        "gen_id": gen_id,
        "run_id": run_id,
        "final_status": "pending",
        "step_1_output": None,
        "step_1_meta": None,
        "step_2_output": None,
        "step_2_meta": None,
        "step_3_meta": None,
        "qc_result": None,
        "qc_attempts": [],
        "error_stage": None,
        "error_message": None,
        "final_image_path": None,
    }


def _mark_failed(record, error_stage, error_message) -> None:
    record["final_status"] = "failed"
    record["error_stage"] = error_stage
    record["error_message"] = error_message
    try:
        _db.finalize_generation(record["gen_id"], "failed", error_message)
    except Exception as e:
        print(f"  [{record['scenario'].get('id', '?')}] DB finalize_generation failed (non-fatal): {e}")


# ──────────────────────────────────────────────────────────────────────────
# Per-scenario processing with pre-loaded pipelines
# ──────────────────────────────────────────────────────────────────────────

def process_scenario_resident(
    record: dict,
    config: dict,
    pipe_1, pipe_2, pipe_3,
    step_3_enabled: bool,
) -> None:
    """
    Run all 3 stages for one scenario using PRE-LOADED pipelines.
    No QC in Ollama flow. Mutates `record` in place.
    """
    scenario = record["scenario"]
    sid = scenario.get("id", "?")
    output_dir = record["output_dir_path"]
    output_dir.mkdir(parents=True, exist_ok=True)
    t_scenario_start = time.time()

    # 1. Save scenario yaml
    try:
        (output_dir / "01_scenario.yaml").write_text(
            json.dumps(scenario, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        _mark_failed(record, "scenario_save", f"failed writing 01_scenario.yaml: {e}")
        return

    # 2. Step 1 prompt (Ollama)
    try:
        step_1_output = _call_with_json_retry(
            lambda: step_1_prompt_builder_ollama.build_step_1_prompt(scenario),
            sid, "Step 1",
        )
    except Exception as e:
        traceback.print_exc()
        try:
            _db.update_step_1(record["gen_id"], status="failed", error=str(e))
        except Exception:
            pass
        _mark_failed(record, "step_1_prompt",
                     f"Step 1 (Ollama) prompt build failed: {type(e).__name__}: {e}")
        return

    step_1_text = (step_1_output or {}).get("step_1_image_prompt", "").strip()
    if not step_1_text:
        try:
            _db.update_step_1(record["gen_id"], status="failed",
                               error="step_1_image_prompt empty in Ollama response")
        except Exception:
            pass
        _mark_failed(record, "step_1_prompt", "step_1_image_prompt empty in Ollama response")
        record["step_1_output"] = step_1_output
        return

    record["step_1_output"] = step_1_output
    (output_dir / "02_step1_prompt.json").write_text(
        json.dumps(step_1_output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    try:
        _db.update_step_1(record["gen_id"], status="prompt_built", prompt=step_1_text)
    except Exception as e:
        print(f"  [{sid}] DB update_step_1 prompt_built failed (non-fatal): {e}")

    # 3. Stage 1 inference (resident pipe_1)
    pulid_params = step_1_output.get("fal_pulid_params") or config.get("step_1", {}).get("defaults", {})
    persona_out_path = output_dir / "03_step1_persona.jpg"
    try:
        step_1_meta = step_1_pulid.generate(
            pipeline=pipe_1,
            step_1_prompt=step_1_text,
            fal_pulid_params=pulid_params,
            out_path=persona_out_path,
            scenario_id=sid,
        )
    except Exception as e:
        traceback.print_exc()
        try:
            _db.update_step_1(record["gen_id"], status="failed",
                               error=f"PuLID Stage 1 failed: {type(e).__name__}: {e}")
        except Exception:
            pass
        _mark_failed(record, "step_1_pulid",
                     f"PuLID Stage 1 failed: {type(e).__name__}: {e}")
        return

    (output_dir / "03_step1_meta.json").write_text(
        json.dumps(step_1_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    record["step_1_meta"] = step_1_meta
    try:
        _db.update_step_1(
            record["gen_id"],
            status="success",
            endpoint=step_1_meta.get("endpoint"),
            image_path=str(persona_out_path),
            request_id=step_1_meta.get("request_id"),
            seed=step_1_meta.get("seed"),
            cost_usd=step_1_meta.get("cost_usd"),
            elapsed_s=step_1_meta.get("elapsed_seconds"),
        )
    except Exception as e:
        print(f"  [{sid}] DB update_step_1 success failed (non-fatal): {e}")

    # 4. Step 2 prompt (Ollama)
    try:
        step_2_output = _call_with_json_retry(
            lambda: step_2_prompt_builder_ollama.build_step_2_prompt(scenario, step_1_output),
            sid, "Step 2",
        )
    except Exception as e:
        traceback.print_exc()
        try:
            _db.update_step_2(record["gen_id"], status="failed", error=str(e))
        except Exception:
            pass
        _mark_failed(record, "step_2_prompt",
                     f"Step 2 (Ollama) prompt build failed: {type(e).__name__}: {e}")
        return

    step_2_text = (step_2_output or {}).get("step_2_image_prompt", "").strip()
    if not step_2_text:
        try:
            _db.update_step_2(record["gen_id"], status="failed",
                               error="step_2_image_prompt empty in Ollama response")
        except Exception:
            pass
        _mark_failed(record, "step_2_prompt", "step_2_image_prompt empty in Ollama response")
        record["step_2_output"] = step_2_output
        return

    record["step_2_output"] = step_2_output
    (output_dir / "04_step2_prompt.json").write_text(
        json.dumps(step_2_output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    try:
        _db.update_step_2(record["gen_id"], status="prompt_built", prompt=step_2_text)
    except Exception as e:
        print(f"  [{sid}] DB update_step_2 prompt_built failed (non-fatal): {e}")

    # 5. Stage 2 inference (resident pipe_2) — NO QC in Ollama flow
    qwen_params = step_2_output.get("fal_qwen_params") or config.get("step_2", {}).get("defaults", {})
    final_out_path = output_dir / "05_step2_final.jpg"
    try:
        step_2_meta = step_2_qwen_edit.generate(
            pipeline=pipe_2,
            step_1_local_path=str(persona_out_path),
            step_2_prompt=step_2_text,
            fal_qwen_params=qwen_params,
            out_path=final_out_path,
            scenario_id=sid,
        )
    except Exception as e:
        traceback.print_exc()
        try:
            _db.update_step_2(record["gen_id"], status="failed",
                               error=f"Qwen Stage 2 failed: {type(e).__name__}: {e}")
        except Exception:
            pass
        _mark_failed(record, "step_2_qwen",
                     f"Qwen Stage 2 failed: {type(e).__name__}: {e}")
        return

    (output_dir / "05_step2_meta.json").write_text(
        json.dumps(step_2_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    record["step_2_meta"] = step_2_meta
    try:
        _db.update_step_2(
            record["gen_id"],
            status="success",
            endpoint=step_2_meta.get("endpoint"),
            image_path=str(final_out_path),
            request_id=step_2_meta.get("request_id"),
            seed=step_2_meta.get("seed"),
            cost_usd=step_2_meta.get("cost_usd"),
            elapsed_s=step_2_meta.get("elapsed_seconds"),
        )
    except Exception as e:
        print(f"  [{sid}] DB update_step_2 success failed (non-fatal): {e}")

    record["final_status"] = "success"
    record["error_message"] = None
    try:
        _db.finalize_generation(record["gen_id"], "success")
    except Exception:
        pass

    # 6. Stage 3 inference (resident pipe_3) — unconditional in Ollama flow (no QC gate)
    if step_3_enabled:
        step_3_out_path = output_dir / "07_step3_realism.jpg"
        lighting_hint = (scenario.get("lighting") or "").strip() or None
        try:
            step_3_meta = step_3_realism.generate(
                pipeline=pipe_3,
                step_2_local_path=str(final_out_path),
                out_path=step_3_out_path,
                scenario_id=sid,
                extra_lighting_hint=lighting_hint,
            )
            (output_dir / "07_step3_meta.json").write_text(
                json.dumps(step_3_meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            record["step_3_meta"] = step_3_meta
            record["final_image_path"] = str(step_3_out_path)
        except Exception as e:
            print(f"  [{sid}] Stage 3 failed (non-fatal — falling back to Stage 2): "
                  f"{type(e).__name__}: {e}")
            traceback.print_exc()
            record["step_3_meta"] = {"error": str(e)}
            record["final_image_path"] = str(final_out_path)
    else:
        record["final_image_path"] = str(final_out_path)
        print(f"  [{sid}] Stage 3 disabled (STEP_3_ENABLED=false)")

    # 7. chain.html — depth: ollama_flow/outputs/<run_id>/<sid>/chain.html
    # → 4 levels up to repo root → assets/persona.jpg
    try:
        trace_html.write_chain_html(
            output_dir, record,
            persona_rel_path="../../../../assets/persona.jpg",
        )
    except Exception as e:
        print(f"  [{sid}] chain.html write failed (non-fatal): {e}")

    scenario_elapsed = time.time() - t_scenario_start
    s1 = (record.get("step_1_meta") or {}).get("elapsed_seconds", 0) or 0
    s2 = (record.get("step_2_meta") or {}).get("elapsed_seconds", 0) or 0
    s3_meta = record.get("step_3_meta")
    s3 = (s3_meta or {}).get("elapsed_seconds", 0) if isinstance(s3_meta, dict) else 0
    print(f"  [{sid}] DONE in {scenario_elapsed:.1f}s  "
          f"(s1={s1:.1f}s, s2={s2:.1f}s, s3={s3:.1f}s, status={record['final_status']})")


# ──────────────────────────────────────────────────────────────────────────
# Summary helpers
# ──────────────────────────────────────────────────────────────────────────

def _build_summary(records, elapsed_s, timestamp, model_label, interrupted) -> dict:
    # No LLM cost in Ollama flow — only GPU cost (tracked externally)
    total_cost = 0.0
    for r in records:
        for stage in ("step_1_meta", "step_2_meta", "step_3_meta"):
            meta = r.get(stage)
            if isinstance(meta, dict) and not meta.get("error"):
                total_cost += float(meta.get("cost_usd") or 0.0)

    succeeded = sum(1 for r in records if r.get("final_status") == "success")
    failed = sum(1 for r in records if r.get("final_status") != "success")

    return {
        "succeeded": succeeded,
        "failed": failed,
        "actual_cost_usd": round(total_cost, 3),
        "elapsed_seconds": elapsed_s,
        "timestamp": timestamp,
        "model_label": model_label,
        "interrupted": interrupted,
    }


def _write_manifest(output_dir, run_id, records, summary, started_at, finished_at, load_seconds) -> None:
    manifest = {
        "run_id": run_id,
        "plan": PLAN_LABEL,
        "started_at": started_at,
        "finished_at": finished_at,
        "load_seconds": load_seconds,
        "elapsed_seconds": int(summary["elapsed_seconds"]),
        **summary,
        "scenarios": [
            {
                "scenario_id": (r.get("scenario") or {}).get("id"),
                "category": (r.get("scenario") or {}).get("category"),
                "archetype": (r.get("scenario") or {}).get("archetype"),
                "difficulty": (r.get("scenario") or {}).get("difficulty"),
                "final_status": r.get("final_status"),
                "error_stage": r.get("error_stage"),
                "error_message": r.get("error_message"),
                "step_1_elapsed_s": (r.get("step_1_meta") or {}).get("elapsed_seconds"),
                "step_2_elapsed_s": (r.get("step_2_meta") or {}).get("elapsed_seconds"),
                "step_3_elapsed_s": (
                    (r.get("step_3_meta") or {}).get("elapsed_seconds")
                    if isinstance(r.get("step_3_meta"), dict) else None
                ),
                "step_3_error": (
                    (r.get("step_3_meta") or {}).get("error")
                    if isinstance(r.get("step_3_meta"), dict) else None
                ),
                "final_image_path": r.get("final_image_path"),
            }
            for r in records
        ],
    }
    (output_dir / "batch_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Alluvi LOCAL image pipeline in ALL-RESIDENT mode with Ollama "
            "(no Anthropic key required, H200 SXM). All 3 pipelines stay resident in "
            "VRAM for the entire batch — zero load overhead, free LLM. Showcase flow."
        )
    )
    parser.add_argument("--scenario", type=str, default=None,
                        help="Run just this one scenario id")
    parser.add_argument("--pilot", action="store_true",
                        help=f"Run first {PILOT_COUNT} scenarios only")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated scenario IDs to include")
    parser.add_argument("--exclude", type=str, default=None,
                        help="Comma-separated scenario IDs to exclude")
    parser.add_argument("--yes", action="store_true",
                        help="Skip cost-confirmation prompt")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip preflight check (not recommended)")
    args = parser.parse_args()

    # 1. Preflight (use ollama_flow's preflight if present, else parent's)
    if not args.skip_preflight:
        preflight_path = OLLAMA_FLOW_ROOT / "preflight_ollama.py"
        if preflight_path.exists():
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("preflight_ollama", preflight_path)
                pf = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(pf)
                if hasattr(pf, "run_preflight"):
                    errors, _warnings = pf.run_preflight(verbose=True)
                    if errors:
                        print("[ollama_resident] preflight failed — aborting")
                        return 1
            except Exception as e:
                print(f"[ollama_resident] preflight crashed: {e} — continuing")
        else:
            try:
                from preflight import run_preflight
                errors, _warnings = run_preflight(verbose=True)
                if errors:
                    print("[ollama_resident] parent preflight failed — aborting")
                    return 1
            except Exception as e:
                print(f"[ollama_resident] no preflight available ({e}) — continuing")

    # 2. Load + filter scenarios
    try:
        all_scenarios = scenario_loader.load_scenarios()
    except Exception as e:
        print(f"[ollama_resident] failed to load scenarios.yaml: {type(e).__name__}: {e}")
        return 1

    scenarios = _filter_scenarios(
        all_scenarios,
        only=args.only, exclude=args.exclude,
        pilot=args.pilot, single=args.scenario,
    )
    if not scenarios:
        print("[ollama_resident] no scenarios after filters — nothing to do")
        return 1

    # 3. Config + cost confirmation
    config = _load_config()
    if not _confirm_cost(len(scenarios), skip=args.yes):
        print("[ollama_resident] aborted by user")
        return 1

    # 4. Output dir + DB run row
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.scenario:
        run_id = f"{timestamp}_{args.scenario}_resident_ollama"
    else:
        pilot_suffix = "_pilot" if args.pilot else ""
        run_id = f"{timestamp}_batch_resident{pilot_suffix}_ollama"
    output_dir = OUTPUT_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    model_label = config.get("model_label",
                              "FLUX+PuLID + Qwen-Edit + Kontext (local, Ollama all-resident H200)")

    notes_parts = [f"{len(scenarios)} scenarios"]
    if args.scenario:
        notes_parts.append(f"single={args.scenario}")
    if args.pilot:
        notes_parts.append("pilot mode")
    if args.only:
        notes_parts.append(f"only={args.only}")
    if args.exclude:
        notes_parts.append(f"exclude={args.exclude}")
    notes = "ollama_resident: " + ", ".join(notes_parts)

    try:
        _db.create_run(run_id=run_id, plan=PLAN_LABEL, pilot_mode=args.pilot, notes=notes)
    except Exception as e:
        print(f"[ollama_resident] DB create_run failed: {e}")
        return 1

    # 5. Initialize records + DB gen rows
    records: list[dict] = []
    for scenario in scenarios:
        sid = scenario.get("id", "?")
        gen_id = uuid.uuid4().hex
        try:
            _db.create_generation(gen_id, run_id, sid, PLAN_LABEL)
        except Exception as e:
            print(f"[ollama_resident] {sid}: DB create_generation failed: {e}")
            continue
        records.append(_init_record(scenario, output_dir / sid, gen_id, run_id, model_label))

    if not records:
        print("[ollama_resident] no records initialized — aborting")
        return 1

    step_3_enabled = os.getenv("STEP_3_ENABLED", "true").lower() == "true"

    # 6. Phase A — pre-load all 3 pipelines
    print("")
    print("=" * 72)
    print(f" ALLUVI — OLLAMA ALL-RESIDENT RUN (H200 SXM, local pipeline)")
    print(f" Run id:    {run_id}")
    print(f" Scenarios: {len(records)}")
    print(f" Output:    {output_dir}")
    print(f" DB:        {_db.DB_PATH}")
    print(f" QC:        disabled (Ollama flow) | Stage 3: {step_3_enabled}")
    print("=" * 72)
    print("")
    print("=" * 72)
    print(f" PHASE A — pre-loading all 3 pipelines into VRAM (~3-5 min)")
    print("=" * 72)

    pipe_1 = pipe_2 = pipe_3 = None
    started_at = datetime.utcnow().isoformat()
    load_start = time.time()

    try:
        vram_utils.reset_vram_peak()

        print(f"  [1/3] loading FLUX-dev + PuLID...")
        pipe_1 = step_1_pulid.load_pipeline()
        vram_utils.report_vram("after FLUX-dev + PuLID load")

        print(f"  [2/3] loading Qwen-Image-Edit-2511...")
        pipe_2 = step_2_qwen_edit.load_pipeline()
        vram_utils.report_vram("after Qwen-Image-Edit-2511 load")

        print(f"  [3/3] loading FLUX.1-Kontext-dev...")
        pipe_3 = step_3_realism.load_pipeline()
        vram_utils.report_vram("after FLUX.1-Kontext-dev load — all 3 resident")

        load_seconds = time.time() - load_start
        print(f"  Phase A done in {load_seconds:.1f}s — all 3 pipelines resident in VRAM")

        # 7. Phase B — process scenarios sequentially
        print("")
        print("=" * 72)
        print(f" PHASE B — processing {len(records)} scenarios (Ollama prompts, zero load overhead)")
        print("=" * 72)
        run_start = time.time()
        interrupted = False
        try:
            for i, record in enumerate(records, start=1):
                sid = record["scenario"].get("id", "?")
                print("")
                print(f"  [{i}/{len(records)}] === {sid} ===")
                process_scenario_resident(
                    record, config, pipe_1, pipe_2, pipe_3,
                    step_3_enabled=step_3_enabled,
                )
        except KeyboardInterrupt:
            interrupted = True
            print("\n  interrupted by user — finalizing partial output...")

        run_elapsed = time.time() - run_start
        print(f"  Phase B done in {run_elapsed:.1f}s ({run_elapsed/60:.1f} min)")

    finally:
        # 8. Phase C — cleanup
        print("")
        print("=" * 72)
        print(f" PHASE C — unloading all pipelines")
        print("=" * 72)
        for name, pipe in (("pipe_3 (Kontext)", pipe_3),
                            ("pipe_2 (Qwen)", pipe_2),
                            ("pipe_1 (PuLID)", pipe_1)):
            if pipe is not None:
                try:
                    print(f"  unloading {name}...")
                    vram_utils.unload_pipeline(pipe)
                except Exception as e:
                    print(f"  {name} unload error (non-fatal): {e}")
        vram_utils.report_vram("after all unloads")

    total_elapsed = time.time() - load_start
    finished_at = datetime.utcnow().isoformat()

    # 9. Write overview + manifest
    summary = _build_summary(records, total_elapsed, timestamp, model_label, False)

    try:
        overview_html.write_overview_html(output_dir, records, summary)
        print(f"  overview.html written → {output_dir / 'overview.html'}")
    except Exception as e:
        print(f"  overview.html write failed (non-fatal): {type(e).__name__}: {e}")
        traceback.print_exc()

    try:
        _write_manifest(output_dir, run_id, records, summary, started_at, finished_at,
                        load_seconds=int(load_seconds))
        print(f"  batch_manifest.json written")
    except Exception as e:
        print(f"  batch_manifest.json write failed (non-fatal): {e}")

    # 10. Finalize DB run
    try:
        _db.finalize_run(
            run_id=run_id,
            total_scenarios=len(records),
            successful=summary["succeeded"],
            failed=summary["failed"],
            total_cost_usd=summary["actual_cost_usd"],
            duration_seconds=int(total_elapsed),
        )
    except Exception as e:
        print(f"[ollama_resident] DB finalize_run failed (non-fatal): {e}")

    # 11. Final summary
    hard_failed = sum(1 for r in records if r.get("final_status") == "failed")
    s3_ran = sum(
        1 for r in records
        if isinstance(r.get("step_3_meta"), dict)
        and not r["step_3_meta"].get("error")
        and r["step_3_meta"].get("local_path")
    )
    s3_failed = sum(
        1 for r in records
        if isinstance(r.get("step_3_meta"), dict) and r["step_3_meta"].get("error")
    )

    print("")
    print("=" * 72)
    print(" OLLAMA ALL-RESIDENT RUN COMPLETE")
    print("=" * 72)
    print(f"  total:           {len(records)}")
    print(f"  ✓ success:       {summary['succeeded']}")
    print(f"  ✗ failed:        {hard_failed}")
    print(f"  cost (LLM):      $0.00  (Ollama runs locally)")
    print(f"  one-time load:   {load_seconds:.1f}s ({load_seconds/60:.1f} min)")
    print(f"  scenario time:   {total_elapsed - load_seconds:.1f}s ({(total_elapsed - load_seconds)/60:.1f} min)")
    print(f"  total wall:      {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    gpu_cost = (total_elapsed / 3600) * POD_HOURLY_USD_H200
    print(f"  gpu cost (H200): ~${gpu_cost:.2f}  ({total_elapsed/3600:.2f}h × ${POD_HOURLY_USD_H200}/hr)")
    if len(records) > 0:
        print(f"  per-scenario:    {(total_elapsed - load_seconds)/len(records):.1f}s avg "
              f"(pure inference, no load overhead)")
    print(f"  model loads:     3 (vs {3 * len(records)} per-scenario)")
    print("")
    print(f"  Stage 3 (Kontext): ran={s3_ran}, failed={s3_failed}")
    print("")
    print(f"  overview.html:   {output_dir / 'overview.html'}")
    print(f"  manifest:        {output_dir / 'batch_manifest.json'}")
    print(f"  DB run_id:       {run_id}")
    print("")

    if hard_failed > 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())