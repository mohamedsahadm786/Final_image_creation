"""
orchestration/stage_batched/run_batch_stage.py — stage-batched runner for the Anthropic flow.

ARCHITECTURE — 5 sequential phases over the full scenario list:

  Phase 1: build all Step 1 prompts (Opus 4.7 calls only, no GPU)
  Phase 2: load FLUX-dev + PuLID once → run Stage 1 for every scenario → unload
  Phase 3: build all Step 2 prompts (Opus 4.7 calls only, no GPU)
  Phase 4: load Qwen-Image-Edit-2511 once → for each eligible scenario,
           run Stage 2 with up to 2 QC retries (Sonnet vision) → unload
  Phase 5: load FLUX.1-Kontext-dev once → run Stage 3 for every QC-passed
           scenario → unload

  Total model loads: 3 (regardless of scenario count).
  vs. per-scenario flow: 3 × N loads (90 for a 30-scenario batch).
  Time saved on a 30-scenario batch: ~30-45 min of pure load/unload overhead.

PROPAGATION RULES:
  - A scenario that fails Phase 1 (Step 1 prompt build) is dropped from
    Phases 2–5.
  - A scenario that fails Phase 2 (Stage 1 inference) is dropped from
    Phases 3–5.
  - A scenario that fails Phase 3 (Step 2 prompt build) is dropped from
    Phases 4–5.
  - A scenario that fails Phase 4 (Stage 2 inference OR QC after 3 attempts)
    is dropped from Phase 5.
  - Phase 5 is non-fatal: a Stage 3 failure falls back to Stage 2 as final.

  At every phase boundary, DB rows are kept in sync so partial-batch state
  survives a crash.

QC retries (Phase 4): the Qwen pipeline stays resident across retries (1
initial + up to 2 retries = 3 inferences max per scenario). Don't unload
between attempts within a single scenario.

CLI (same as per-scenario run_batch.py):
    python orchestration/stage_batched/run_batch_stage.py
    python orchestration/stage_batched/run_batch_stage.py --pilot
    python orchestration/stage_batched/run_batch_stage.py --only ID1,ID2
    python orchestration/stage_batched/run_batch_stage.py --yes --skip-preflight
"""

import argparse
import json
import os
import shutil
import sys
import time
import traceback
import uuid
import yaml
from datetime import datetime
from pathlib import Path

# Repo root: walk up from orchestration/stage_batched/run_batch_stage.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import db
from src import scenario_loader
from src import step_1_prompt_builder
from src import step_1_pulid
from src import step_2_prompt_builder
from src import step_2_qwen_edit
from src import step_3_realism
from src import trace_html
from src import overview_html
from src import vram_utils
from src.json_utils import JSONSanityError


CONFIG_PATH = REPO_ROOT / "config.yaml"
OUTPUT_ROOT = REPO_ROOT / "outputs"

PLAN_LABEL = "pulid_qwen_kontext_local_stage_batched"

PILOT_COUNT = 5
MAX_JSON_RETRIES = 1            # Opus is not free — cap retries low
MAX_QC_RETRIES = 2              # 1 initial + 2 retries = 3 Qwen calls max per scenario
POD_HOURLY_USD = 1.89

# Estimated wall time per scenario in stage-batched mode (lower than per-scenario
# because model loads happen only 3 times for the whole batch, not 3 × N).
DEFAULT_WALL_TIME_PER_SCENARIO_S = 240  # ~4 min average


# ──────────────────────────────────────────────────────────────────────────
# Config + helpers
# ──────────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"missing config: {CONFIG_PATH}")
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _call_with_json_retry(build_fn, scenario_id: str, step_label: str, max_retries: int = MAX_JSON_RETRIES):
    """Call build_fn() and retry only on JSONSanityError."""
    last_error = None
    for attempt in range(1, max_retries + 2):
        try:
            return build_fn()
        except JSONSanityError as e:
            last_error = e
            if attempt <= max_retries:
                print(
                    f"  {scenario_id} {step_label}: JSON sanity error on attempt "
                    f"{attempt}/{max_retries + 1}: {str(e)[:200]}"
                )
            else:
                print(
                    f"  {scenario_id} {step_label}: all {max_retries + 1} "
                    f"attempts failed JSON sanity check"
                )
    raise last_error


def _filter_scenarios(all_scenarios, only=None, exclude=None, pilot=False):
    """Apply --only / --exclude / --pilot."""
    if only:
        only_set = {x.strip() for x in only.split(",") if x.strip()}
        filtered = [s for s in all_scenarios if s.get("id") in only_set]
        missing = only_set - {s.get("id") for s in filtered}
        if missing:
            print(f"[batch_stage] WARNING: --only IDs not found: {sorted(missing)}")
    else:
        filtered = list(all_scenarios)
    if exclude:
        exclude_set = {x.strip() for x in exclude.split(",") if x.strip()}
        filtered = [s for s in filtered if s.get("id") not in exclude_set]
    if pilot:
        filtered = filtered[:PILOT_COUNT]
    return filtered


def _confirm_cost(n_scenarios, cost_per_scenario, skip=False) -> bool:
    llm_cost_total = n_scenarios * cost_per_scenario
    wall_seconds = n_scenarios * DEFAULT_WALL_TIME_PER_SCENARIO_S
    wall_hours = wall_seconds / 3600
    gpu_cost_total = wall_hours * POD_HOURLY_USD

    print("")
    print("=" * 72)
    print(" BATCH PLAN (stage-batched — 3 model loads total)")
    print("=" * 72)
    print(f"  scenarios:           {n_scenarios}")
    print(f"  LLM cost (est):      ~${llm_cost_total:.2f}  (~${cost_per_scenario:.2f}/scenario)")
    print(f"  Wall time (est):     ~{wall_seconds/60:.0f} min  (~{DEFAULT_WALL_TIME_PER_SCENARIO_S}s/scenario)")
    print(f"  GPU cost (est):      ~${gpu_cost_total:.2f}  ({wall_hours:.2f}h × ${POD_HOURLY_USD}/hr)")
    print(f"  Combined (est):      ~${llm_cost_total + gpu_cost_total:.2f}")
    print("=" * 72)
    print("")

    if skip:
        print("[batch_stage] --yes flag: skipping confirmation")
        return True
    try:
        response = input("Proceed? [y/N]: ").strip().lower()
    except EOFError:
        response = ""
    return response in ("y", "yes")


# ──────────────────────────────────────────────────────────────────────────
# Record helpers
# ──────────────────────────────────────────────────────────────────────────

def _init_record(scenario: dict, output_dir: Path, gen_id: str, run_id: str, model_label: str) -> dict:
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


def _mark_failed(record: dict, error_stage: str, error_message: str) -> None:
    record["final_status"] = "failed"
    record["error_stage"] = error_stage
    record["error_message"] = error_message
    try:
        db.finalize_generation(record["gen_id"], "failed", error_message)
    except Exception as e:
        print(f"  [{record['scenario'].get('id', '?')}] DB finalize_generation failed (non-fatal): {e}")


def _scenario_eligible(record: dict) -> bool:
    """True if scenario hasn't failed yet — should continue to the next phase."""
    return record["final_status"] not in ("failed", "qc_failed")


# ──────────────────────────────────────────────────────────────────────────
# Phase 1 — build all Step 1 prompts (Opus, no GPU)
# ──────────────────────────────────────────────────────────────────────────

def phase_1_step_1_prompts(records: list[dict]) -> None:
    print("")
    print("=" * 72)
    print(f" PHASE 1 — Step 1 prompt build (Opus 4.7) for {len(records)} scenarios")
    print("=" * 72)
    t0 = time.time()
    for i, record in enumerate(records, start=1):
        scenario = record["scenario"]
        sid = scenario.get("id", "?")
        output_dir = record["output_dir_path"]
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save scenario yaml
        try:
            (output_dir / "01_scenario.yaml").write_text(
                json.dumps(scenario, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as e:
            _mark_failed(record, "scenario_save", f"failed writing 01_scenario.yaml: {e}")
            continue

        print(f"  [{i}/{len(records)}] {sid}: building Step 1 prompt...")
        try:
            step_1_output = _call_with_json_retry(
                lambda: step_1_prompt_builder.build_step_1_prompt(scenario),
                sid, "Step 1",
            )
        except Exception as e:
            traceback.print_exc()
            try:
                db.update_step_1(record["gen_id"], status="failed", error=str(e))
            except Exception:
                pass
            _mark_failed(record, "step_1_prompt",
                         f"Step 1 prompt build failed: {type(e).__name__}: {e}")
            continue

        step_1_text = (step_1_output or {}).get("step_1_image_prompt", "").strip()
        if not step_1_text:
            try:
                db.update_step_1(record["gen_id"], status="failed",
                                  error="step_1_image_prompt empty in Opus response")
            except Exception:
                pass
            _mark_failed(record, "step_1_prompt", "step_1_image_prompt empty in Opus response")
            record["step_1_output"] = step_1_output
            continue

        record["step_1_output"] = step_1_output
        (output_dir / "02_step1_prompt.json").write_text(
            json.dumps(step_1_output, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        try:
            db.update_step_1(record["gen_id"], status="prompt_built", prompt=step_1_text)
        except Exception as e:
            print(f"  [{sid}] DB update_step_1 prompt_built failed (non-fatal): {e}")

    elapsed = time.time() - t0
    succeeded = sum(1 for r in records if r.get("step_1_output"))
    print(f"  Phase 1 done in {elapsed:.1f}s: {succeeded}/{len(records)} prompts built")


# ──────────────────────────────────────────────────────────────────────────
# Phase 2 — Stage 1 PuLID inference (single load, all eligible scenarios)
# ──────────────────────────────────────────────────────────────────────────

def phase_2_stage_1_inference(records: list[dict], config: dict) -> None:
    eligible = [r for r in records if _scenario_eligible(r) and r.get("step_1_output")]

    print("")
    print("=" * 72)
    print(f" PHASE 2 — Stage 1 (FLUX-dev + PuLID) for {len(eligible)} scenarios")
    print("=" * 72)

    if not eligible:
        print("  no eligible scenarios — skipping Phase 2")
        return

    pipeline = None
    try:
        vram_utils.reset_vram_peak()
        pipeline = step_1_pulid.load_pipeline()
        vram_utils.report_vram("phase 2: stage 1 loaded")

        t0 = time.time()
        for i, record in enumerate(eligible, start=1):
            scenario = record["scenario"]
            sid = scenario.get("id", "?")
            output_dir = record["output_dir_path"]
            step_1_output = record["step_1_output"]
            step_1_text = step_1_output["step_1_image_prompt"]

            pulid_params = step_1_output.get("fal_pulid_params") or config.get(
                "step_1", {}
            ).get("defaults", {})
            persona_out_path = output_dir / "03_step1_persona.jpg"

            print(f"  [{i}/{len(eligible)}] {sid}: inferring Stage 1...")
            try:
                step_1_meta = step_1_pulid.generate(
                    pipeline=pipeline,
                    step_1_prompt=step_1_text,
                    fal_pulid_params=pulid_params,
                    out_path=persona_out_path,
                    scenario_id=sid,
                )
            except Exception as e:
                traceback.print_exc()
                try:
                    db.update_step_1(record["gen_id"], status="failed",
                                      error=f"PuLID Stage 1 failed: {type(e).__name__}: {e}")
                except Exception:
                    pass
                _mark_failed(record, "step_1_pulid",
                             f"PuLID Stage 1 failed: {type(e).__name__}: {e}")
                continue

            (output_dir / "03_step1_meta.json").write_text(
                json.dumps(step_1_meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            record["step_1_meta"] = step_1_meta
            try:
                db.update_step_1(
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

        elapsed = time.time() - t0
        succeeded = sum(1 for r in eligible if r.get("step_1_meta"))
        print(f"  Phase 2 done in {elapsed:.1f}s: {succeeded}/{len(eligible)} Stage 1 inferences succeeded")
    finally:
        if pipeline is not None:
            vram_utils.unload_pipeline(pipeline)
            vram_utils.report_vram("phase 2: stage 1 unloaded")


# ──────────────────────────────────────────────────────────────────────────
# Phase 3 — build all Step 2 prompts (Opus, no GPU)
# ──────────────────────────────────────────────────────────────────────────

def phase_3_step_2_prompts(records: list[dict]) -> None:
    eligible = [r for r in records if _scenario_eligible(r) and r.get("step_1_meta")]

    print("")
    print("=" * 72)
    print(f" PHASE 3 — Step 2 prompt build (Opus 4.7) for {len(eligible)} scenarios")
    print("=" * 72)

    if not eligible:
        print("  no eligible scenarios — skipping Phase 3")
        return

    t0 = time.time()
    for i, record in enumerate(eligible, start=1):
        scenario = record["scenario"]
        sid = scenario.get("id", "?")
        output_dir = record["output_dir_path"]
        step_1_output = record["step_1_output"]

        print(f"  [{i}/{len(eligible)}] {sid}: building Step 2 prompt...")
        try:
            step_2_output = _call_with_json_retry(
                lambda: step_2_prompt_builder.build_step_2_prompt(scenario, step_1_output),
                sid, "Step 2",
            )
        except Exception as e:
            traceback.print_exc()
            try:
                db.update_step_2(record["gen_id"], status="failed", error=str(e))
            except Exception:
                pass
            _mark_failed(record, "step_2_prompt",
                         f"Step 2 prompt build failed: {type(e).__name__}: {e}")
            continue

        step_2_text = (step_2_output or {}).get("step_2_image_prompt", "").strip()
        if not step_2_text:
            try:
                db.update_step_2(record["gen_id"], status="failed",
                                  error="step_2_image_prompt empty in Opus response")
            except Exception:
                pass
            _mark_failed(record, "step_2_prompt", "step_2_image_prompt empty in Opus response")
            record["step_2_output"] = step_2_output
            continue

        record["step_2_output"] = step_2_output
        (output_dir / "04_step2_prompt.json").write_text(
            json.dumps(step_2_output, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        try:
            db.update_step_2(record["gen_id"], status="prompt_built", prompt=step_2_text)
        except Exception as e:
            print(f"  [{sid}] DB update_step_2 prompt_built failed (non-fatal): {e}")

    elapsed = time.time() - t0
    succeeded = sum(1 for r in eligible if r.get("step_2_output"))
    print(f"  Phase 3 done in {elapsed:.1f}s: {succeeded}/{len(eligible)} prompts built")


# ──────────────────────────────────────────────────────────────────────────
# Phase 4 — Stage 2 Qwen inference + QC retries (single load)
# ──────────────────────────────────────────────────────────────────────────

def phase_4_stage_2_inference(records: list[dict], config: dict) -> None:
    eligible = [r for r in records if _scenario_eligible(r) and r.get("step_2_output")]
    qc_enabled = os.getenv("QC_ENABLED", "true").lower() == "true"

    print("")
    print("=" * 72)
    print(f" PHASE 4 — Stage 2 (Qwen-Image-Edit-2511) + QC for {len(eligible)} scenarios")
    print(f"           qc_enabled={qc_enabled}, max_qc_retries={MAX_QC_RETRIES}")
    print("=" * 72)

    if not eligible:
        print("  no eligible scenarios — skipping Phase 4")
        return

    pipeline = None
    try:
        vram_utils.reset_vram_peak()
        pipeline = step_2_qwen_edit.load_pipeline()
        vram_utils.report_vram("phase 4: stage 2 loaded")

        # Import QC validator lazily (inside the phase) — it's only used here.
        from src.qc_validator import validate_image

        t0 = time.time()
        for i, record in enumerate(eligible, start=1):
            scenario = record["scenario"]
            sid = scenario.get("id", "?")
            output_dir = record["output_dir_path"]
            step_2_output = record["step_2_output"]
            step_2_text = step_2_output["step_2_image_prompt"]
            persona_path = output_dir / "03_step1_persona.jpg"
            final_out_path = output_dir / "05_step2_final.jpg"

            qwen_params = step_2_output.get("fal_qwen_params") or config.get(
                "step_2", {}
            ).get("defaults", {})

            print(f"  [{i}/{len(eligible)}] {sid}: Stage 2 with QC...")

            final_qc_result = None
            final_step_2_meta = None
            qc_attempts: list[dict] = []

            for attempt in range(1, MAX_QC_RETRIES + 2):  # 1, 2, 3
                attempt_image_path = output_dir / f"05_step2_final_attempt_{attempt}.jpg"
                is_last = attempt == MAX_QC_RETRIES + 1

                # Stage 2 inference
                try:
                    step_2_meta = step_2_qwen_edit.generate(
                        pipeline=pipeline,
                        step_1_local_path=str(persona_path),
                        step_2_prompt=step_2_text,
                        fal_qwen_params=qwen_params,
                        out_path=attempt_image_path,
                        scenario_id=f"{sid}#a{attempt}",
                    )
                except Exception as e:
                    traceback.print_exc()
                    try:
                        db.update_step_2(record["gen_id"], status="failed",
                                          error=f"Qwen Stage 2 failed on attempt {attempt}: {e}")
                    except Exception:
                        pass
                    _mark_failed(record, "step_2_qwen",
                                 f"Qwen Stage 2 failed on attempt {attempt}: {type(e).__name__}: {e}")
                    final_step_2_meta = None
                    break

                # Copy this attempt to the canonical filename
                try:
                    shutil.copy(attempt_image_path, final_out_path)
                except Exception as e:
                    print(f"  [{sid}] copy attempt image failed (non-fatal): {e}")

                final_step_2_meta = step_2_meta

                # QC (or skip)
                if not qc_enabled:
                    print(f"  [{sid}] QC disabled, accepting attempt {attempt}")
                    final_qc_result = {
                        "passed": True, "score": None, "issues": [],
                        "recommendation": "use", "error": "QC_ENABLED=false",
                    }
                    break

                try:
                    qc_result = validate_image(final_out_path, scenario_id=f"{sid}#a{attempt}")
                except Exception as e:
                    print(f"  [{sid}] QC crashed on attempt {attempt}: {e}")
                    qc_result = {
                        "passed": True,  # treat infra failure as pass
                        "score": 0.5,
                        "issues": [f"QC crashed: {e}"],
                        "recommendation": "use",
                        "error": str(e),
                    }

                (output_dir / f"06_qc_result_attempt_{attempt}.json").write_text(
                    json.dumps(qc_result, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                qc_attempts.append({
                    "attempt": attempt,
                    "passed": qc_result.get("passed"),
                    "score": qc_result.get("score"),
                    "issues": qc_result.get("issues", []),
                })

                if qc_result.get("passed"):
                    print(f"  [{sid}] QC PASSED on attempt {attempt}")
                    final_qc_result = qc_result
                    break

                if is_last:
                    print(f"  [{sid}] QC failed on final attempt {attempt} — skipping Stage 3")
                    final_qc_result = qc_result
                else:
                    print(f"  [{sid}] QC failed attempt {attempt}/{MAX_QC_RETRIES + 1} — retrying Stage 2")

            # If hard error broke the loop and we never got step_2_meta, scenario already marked failed
            if final_step_2_meta is None:
                continue

            # Save final canonical step_2_meta + qc result
            (output_dir / "05_step2_meta.json").write_text(
                json.dumps(final_step_2_meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            record["step_2_meta"] = final_step_2_meta
            if final_qc_result:
                (output_dir / "06_qc_result.json").write_text(
                    json.dumps({**final_qc_result, "attempts": qc_attempts},
                               indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                record["qc_result"] = final_qc_result
                record["qc_attempts"] = qc_attempts

            try:
                db.update_step_2(
                    record["gen_id"],
                    status="success",
                    endpoint=final_step_2_meta.get("endpoint"),
                    image_path=str(final_out_path),
                    request_id=final_step_2_meta.get("request_id"),
                    seed=final_step_2_meta.get("seed"),
                    cost_usd=final_step_2_meta.get("cost_usd"),
                    elapsed_s=final_step_2_meta.get("elapsed_seconds"),
                )
            except Exception as e:
                print(f"  [{sid}] DB update_step_2 success failed (non-fatal): {e}")

            # Outcome for this scenario
            if final_qc_result and final_qc_result.get("passed"):
                record["final_status"] = "success"
                record["error_message"] = None
                try:
                    db.finalize_generation(record["gen_id"], "success")
                except Exception:
                    pass
            else:
                record["final_status"] = "qc_failed"
                issues = "; ".join((final_qc_result or {}).get("issues", []) or ["QC failed"])
                record["error_stage"] = "qc"
                record["error_message"] = f"QC failed after {MAX_QC_RETRIES + 1} attempts: {issues}"
                try:
                    db.finalize_generation(record["gen_id"], "qc_failed", record["error_message"])
                except Exception:
                    pass

            # Optional db.update_qc (graceful if missing)
            if final_qc_result is not None:
                try:
                    db.update_qc(
                        record["gen_id"],
                        passed=final_qc_result.get("passed", False),
                        score=final_qc_result.get("score", 0.0),
                        issues=final_qc_result.get("issues", []),
                    )
                except AttributeError:
                    pass
                except Exception as e:
                    print(f"  [{sid}] db.update_qc failed (non-fatal): {e}")

        elapsed = time.time() - t0
        succeeded = sum(1 for r in eligible if r.get("final_status") == "success")
        qc_failed = sum(1 for r in eligible if r.get("final_status") == "qc_failed")
        print(f"  Phase 4 done in {elapsed:.1f}s: {succeeded} passed QC, {qc_failed} failed QC")
    finally:
        if pipeline is not None:
            vram_utils.unload_pipeline(pipeline)
            vram_utils.report_vram("phase 4: stage 2 unloaded")


# ──────────────────────────────────────────────────────────────────────────
# Phase 5 — Stage 3 Kontext refinement (single load, QC-passed only)
# ──────────────────────────────────────────────────────────────────────────

def phase_5_stage_3_inference(records: list[dict]) -> None:
    step_3_enabled = os.getenv("STEP_3_ENABLED", "true").lower() == "true"

    # Eligible: scenario succeeded through Stage 2 AND QC passed
    eligible = [
        r for r in records
        if r.get("final_status") == "success"
        and bool((r.get("qc_result") or {}).get("passed"))
        and r.get("step_2_meta")
    ]

    print("")
    print("=" * 72)
    print(f" PHASE 5 — Stage 3 (FLUX.1-Kontext-dev) for {len(eligible)} QC-passed scenarios")
    print(f"           step_3_enabled={step_3_enabled}")
    print("=" * 72)

    # Even if not eligible or disabled, set the canonical final_image_path for every record
    if not step_3_enabled or not eligible:
        for r in records:
            if r.get("step_2_meta"):
                r["final_image_path"] = str(r["output_dir_path"] / "05_step2_final.jpg")
        if not step_3_enabled:
            print(f"  Stage 3 disabled (STEP_3_ENABLED=false) — skipping Phase 5")
        else:
            print(f"  no eligible scenarios — skipping Phase 5")
        return

    pipeline = None
    try:
        vram_utils.reset_vram_peak()
        pipeline = step_3_realism.load_pipeline()
        vram_utils.report_vram("phase 5: stage 3 loaded")

        t0 = time.time()
        for i, record in enumerate(eligible, start=1):
            scenario = record["scenario"]
            sid = scenario.get("id", "?")
            output_dir = record["output_dir_path"]
            final_out_path = output_dir / "05_step2_final.jpg"
            step_3_out_path = output_dir / "07_step3_realism.jpg"
            lighting_hint = (scenario.get("lighting") or "").strip() or None

            print(f"  [{i}/{len(eligible)}] {sid}: Stage 3 realism...")
            try:
                step_3_meta = step_3_realism.generate(
                    pipeline=pipeline,
                    step_2_local_path=str(final_out_path),
                    out_path=step_3_out_path,
                    scenario_id=sid,
                    extra_lighting_hint=lighting_hint,
                )
                (output_dir / "07_step3_meta.json").write_text(
                    json.dumps(step_3_meta, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                record["step_3_meta"] = step_3_meta
                record["final_image_path"] = str(step_3_out_path)
            except Exception as e:
                print(f"  [{sid}] Stage 3 failed (non-fatal — falling back to Stage 2): "
                      f"{type(e).__name__}: {e}")
                traceback.print_exc()
                record["step_3_meta"] = {"error": str(e)}
                record["final_image_path"] = str(final_out_path)

        # For QC-failed / Stage-1-failed records, set final_image_path to whatever
        # they did produce (or None)
        for r in records:
            if r.get("final_image_path"):
                continue
            if r.get("step_2_meta"):
                r["final_image_path"] = str(r["output_dir_path"] / "05_step2_final.jpg")

        elapsed = time.time() - t0
        s3_ran = sum(
            1 for r in eligible
            if isinstance(r.get("step_3_meta"), dict) and not r["step_3_meta"].get("error")
        )
        s3_failed = sum(
            1 for r in eligible
            if isinstance(r.get("step_3_meta"), dict) and r["step_3_meta"].get("error")
        )
        print(f"  Phase 5 done in {elapsed:.1f}s: {s3_ran} succeeded, {s3_failed} failed (non-fatal)")
    finally:
        if pipeline is not None:
            vram_utils.unload_pipeline(pipeline)
            vram_utils.report_vram("phase 5: stage 3 unloaded")


# ──────────────────────────────────────────────────────────────────────────
# Phase 6 — write per-scenario chain.html + batch overview/manifest
# ──────────────────────────────────────────────────────────────────────────

def _build_summary_for_overview(records, elapsed_s, config, timestamp, model_label, interrupted) -> dict:
    """Build the summary dict in the exact shape src/overview_html.write_overview_html expects."""
    c_opus_1 = config.get("step_1", {}).get("cost_per_prompt_opus_usd", 0.10)
    c_opus_2 = config.get("step_2", {}).get("cost_per_prompt_opus_usd", 0.18)

    actual_cost = 0.0
    for r in records:
        for stage in ("step_1_meta", "step_2_meta", "step_3_meta"):
            meta = r.get(stage)
            if isinstance(meta, dict) and not meta.get("error"):
                actual_cost += float(meta.get("cost_usd") or 0.0)
        if r.get("step_1_output"):
            actual_cost += c_opus_1
        if r.get("step_2_output"):
            actual_cost += c_opus_2

    succeeded = sum(1 for r in records if r.get("final_status") == "success")
    failed = sum(1 for r in records if r.get("final_status") != "success")

    return {
        "succeeded": succeeded,
        "failed": failed,
        "actual_cost_usd": round(actual_cost, 3),
        "elapsed_seconds": elapsed_s,
        "timestamp": timestamp,
        "model_label": model_label,
        "interrupted": interrupted,
    }


def phase_6_html_and_manifest(
    output_dir: Path, records: list[dict], started_at: str, finished_at: str,
    elapsed_s: float, config: dict, timestamp: str, model_label: str, interrupted: bool,
    run_id: str,
) -> dict:
    print("")
    print("=" * 72)
    print(f" PHASE 6 — writing chain.html × {len(records)} + overview.html + manifest")
    print("=" * 72)

    # chain.html per scenario. Batch layout depth: outputs/<ts>_batch_stage/<sid>/chain.html
    # → 3 levels up to repo root → assets/persona.jpg
    for record in records:
        sid = record["scenario"].get("id", "?")
        try:
            trace_html.write_chain_html(
                record["output_dir_path"], record,
                persona_rel_path="../../../assets/persona.jpg",
            )
        except Exception as e:
            print(f"  [{sid}] chain.html write failed (non-fatal): {e}")

    # Build summary in overview_html's shape
    summary = _build_summary_for_overview(
        records, elapsed_s, config, timestamp, model_label, interrupted,
    )

    # overview.html — direct call (signature is now known)
    try:
        overview_html.write_overview_html(output_dir, records, summary)
        print(f"  overview.html written → {output_dir / 'overview.html'}")
    except Exception as e:
        print(f"  overview.html write failed (non-fatal): {type(e).__name__}: {e}")
        traceback.print_exc()

    # batch_manifest.json
    manifest = {
        "run_id": run_id,
        "plan": PLAN_LABEL,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": int(elapsed_s),
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
                "qc_passed": (r.get("qc_result") or {}).get("passed"),
                "qc_score": (r.get("qc_result") or {}).get("score"),
                "qc_attempts": len(r.get("qc_attempts") or []),
                "final_image_path": r.get("final_image_path"),
            }
            for r in records
        ],
    }
    try:
        (output_dir / "batch_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  batch_manifest.json written → {output_dir / 'batch_manifest.json'}")
    except Exception as e:
        print(f"  batch_manifest.json write failed (non-fatal): {e}")

    return summary


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Alluvi LOCAL image generation pipeline in STAGE-BATCHED mode "
            "(all scenarios go through Stage 1, then all through Stage 2 + QC, "
            "then all QC-passed through Stage 3 — 3 model loads total)."
        )
    )
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

    # 1. Preflight
    if not args.skip_preflight:
        from preflight import run_preflight
        errors, _warnings = run_preflight(verbose=True)
        if errors:
            print("[batch_stage] preflight failed — aborting batch")
            return 1

    # 2. Load + filter scenarios
    try:
        all_scenarios = scenario_loader.load_scenarios()
    except Exception as e:
        print(f"[batch_stage] failed to load scenarios.yaml: {type(e).__name__}: {e}")
        return 1

    scenarios = _filter_scenarios(
        all_scenarios, only=args.only, exclude=args.exclude, pilot=args.pilot
    )
    if not scenarios:
        print("[batch_stage] no scenarios after filters — nothing to do")
        return 1

    # 3. Config + cost confirmation
    config = _load_config()
    cost_per = float(config.get("cost_per_scenario_usd", 0.30))
    if not _confirm_cost(len(scenarios), cost_per, skip=args.yes):
        print("[batch_stage] aborted by user")
        return 1

    # 4. Output dir + DB run row
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pilot_suffix = "_pilot" if args.pilot else ""
    run_id = f"{timestamp}_batch_stage{pilot_suffix}"
    output_dir = OUTPUT_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    model_label = config.get("model_label",
                              "FLUX+PuLID + Qwen-Image-Edit-2511 + FLUX.1-Kontext-dev (local, stage-batched)")

    notes_parts = [f"{len(scenarios)} scenarios"]
    if args.pilot:
        notes_parts.append("pilot mode")
    if args.only:
        notes_parts.append(f"only={args.only}")
    if args.exclude:
        notes_parts.append(f"exclude={args.exclude}")
    notes = "batch_stage: " + ", ".join(notes_parts)

    try:
        db.create_run(run_id=run_id, plan=PLAN_LABEL, pilot_mode=args.pilot, notes=notes)
    except Exception as e:
        print(f"[batch_stage] DB create_run failed: {e}")
        return 1

    # 5. Create per-scenario gen rows + record dicts
    records: list[dict] = []
    for scenario in scenarios:
        sid = scenario.get("id", "?")
        gen_id = uuid.uuid4().hex
        try:
            db.create_generation(gen_id, run_id, sid, PLAN_LABEL)
        except Exception as e:
            print(f"[batch_stage] {sid}: DB create_generation failed: {e}")
            continue
        record = _init_record(scenario, output_dir / sid, gen_id, run_id, model_label)
        records.append(record)

    if not records:
        print("[batch_stage] no records initialized — aborting")
        return 1

    # 6. Run phases
    print("")
    print("=" * 72)
    print(f" ALLUVI — STAGE-BATCHED RUN (Anthropic flow, local pipeline)")
    print(f" Run id:    {run_id}")
    print(f" Scenarios: {len(records)}")
    print(f" Output:    {output_dir}")
    print("=" * 72)

    started_at = datetime.utcnow().isoformat()
    started = time.time()
    interrupted = False

    try:
        phase_1_step_1_prompts(records)
        phase_2_stage_1_inference(records, config)
        phase_3_step_2_prompts(records)
        phase_4_stage_2_inference(records, config)
        phase_5_stage_3_inference(records)
    except KeyboardInterrupt:
        interrupted = True
        print("\n[batch_stage] interrupted by user — writing partial output...")

    elapsed = time.time() - started
    finished_at = datetime.utcnow().isoformat()

    # 7. Phase 6 — html + manifest
    summary = phase_6_html_and_manifest(
        output_dir, records, started_at, finished_at, elapsed,
        config, timestamp, model_label, interrupted, run_id,
    )

    # 8. Finalize DB run
    try:
        db.finalize_run(
            run_id=run_id,
            total_scenarios=len(records),
            successful=summary["succeeded"],
            failed=summary["failed"],
            total_cost_usd=summary["actual_cost_usd"],
            duration_seconds=int(elapsed),
        )
    except Exception as e:
        print(f"[batch_stage] DB finalize_run failed (non-fatal): {e}")

    # 9. Final summary print
    qc_failed = sum(1 for r in records if r.get("final_status") == "qc_failed")
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
    print(" BATCH COMPLETE" + (" (interrupted)" if interrupted else ""))
    print("=" * 72)
    print(f"  total:        {len(records)}")
    print(f"  ✓ success:    {summary['succeeded']}")
    print(f"  ⚠ qc failed:  {qc_failed}")
    print(f"  ✗ failed:     {hard_failed}")
    print(f"  cost:         ${summary['actual_cost_usd']:.2f}  (LLM only — GPU separate)")
    print(f"  wall time:    {elapsed/60:.1f} min")
    gpu_cost = (elapsed / 3600) * POD_HOURLY_USD
    print(f"  gpu cost:     ~${gpu_cost:.2f}  ({elapsed/3600:.2f}h × ${POD_HOURLY_USD}/hr)")
    print(f"  model loads:  3 (vs {3 * len(records)} in per-scenario flow)")
    print("")
    print(f"  Stage 3 (Kontext): ran={s3_ran}, failed={s3_failed}")
    print("")
    print(f"  overview.html: {output_dir / 'overview.html'}")
    print(f"  manifest:      {output_dir / 'batch_manifest.json'}")
    print(f"  DB run_id:     {run_id}")
    print("")

    if interrupted:
        return 130
    if hard_failed > 0:
        return 2
    if qc_failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())