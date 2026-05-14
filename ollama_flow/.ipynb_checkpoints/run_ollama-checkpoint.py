"""
ollama_flow/run_ollama.py — single-scenario runner for Ollama mode (LOCAL stages).

Replicates orchestration/per_scenario/run.py's flow but:
  - Uses Ollama-mode prompt builders from ollama_flow/ollama_src/
  - Routes the SQLite DB to ollama_flow/data/alluvi_ollama.db
    (SEPARATE from production alluvi.db — iteration runs don't pollute
    production evaluation queries)
  - Outputs to ollama_flow/outputs/ (separate from parent's outputs/)
  - Reads ollama_flow/config.yaml (separate from parent's config.yaml)
  - PLAN_LABEL='pulid_qwen_tuned_ollama' so DB rows are distinguishable
  - NO QC step (Ollama flow runs Stage 3 unconditionally after a successful
    Stage 2 — no Sonnet validator means no automatic gate)

LLM cost: $0.00 (Ollama runs locally)
GPU wall time: ~3-5 min per scenario (3 model loads + 3 inferences)
Peak VRAM: ~48 GB (Stage 2 Qwen)

Usage (from inside ollama_flow/):
    python run_ollama.py --scenario travel_hotel_morning_29

Why the local package is named `ollama_src` (not `src`):
    Parent repo already has src/ — naming our folder ollama_src/ avoids
    sys.path shadowing. Both packages coexist cleanly.
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


# Path setup — ollama_flow lives at <repo>/ollama_flow/
OLLAMA_FLOW_ROOT = Path(__file__).resolve().parent
PARENT_REPO_ROOT = OLLAMA_FLOW_ROOT.parent


# Put BOTH on sys.path. Parent goes in LAST so it ends up at sys.path[0]
# and wins any name lookup. (sys.path.insert(0, x) prepends.)
# Result:
#   `from src import db`        → PARENT/src/db.py ✓
#   `from ollama_src import X`  → OLLAMA_FLOW_ROOT/ollama_src/X.py ✓
sys.path.insert(0, str(OLLAMA_FLOW_ROOT))
sys.path.insert(0, str(PARENT_REPO_ROOT))


# ─── Redirect DB to ollama_flow's data/ folder BEFORE any DB calls ──────
# Monkey-patch db.DB_PATH so Ollama runs land in ollama_flow/data/alluvi_ollama.db,
# NOT the parent's data/alluvi.db. Production DB stays untouched.
from src import db as _db
_db.DB_PATH = OLLAMA_FLOW_ROOT / "data" / "alluvi_ollama.db"


# ─── Parent's infrastructure modules (shared with production) ───────────
from src import scenario_loader
from src import step_1_pulid          # local FLUX-dev + PuLID
from src import step_2_qwen_edit      # local Qwen-Image-Edit-2511
from src import step_3_realism        # local FLUX.1-Kontext-dev
from src import trace_html
from src import vram_utils            # load/unload helper


# ─── Our own Ollama prompt builders ─────────────────────────────────────
from ollama_src import step_1_prompt_builder_ollama
from ollama_src import step_2_prompt_builder_ollama


# ─── Constants ────────────────────────────────────────────────────────────
CONFIG_PATH = OLLAMA_FLOW_ROOT / "config.yaml"
OUTPUT_ROOT = OLLAMA_FLOW_ROOT / "outputs"

PLAN_LABEL = "pulid_qwen_tuned_ollama"


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"missing config: {CONFIG_PATH}")
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


# ─── JSON retry config ────────────────────────────────────────────────────
# Ollama (especially 7B models) occasionally returns malformed JSON.
# Retry the SAME scenario up to MAX_JSON_RETRIES times. Cheap because
# Ollama is free.
MAX_JSON_RETRIES = 2  # initial + 2 retries = up to 3 total attempts


def _call_with_json_retry(
    build_fn,
    scenario_id: str,
    step_label: str,
    max_retries: int = MAX_JSON_RETRIES,
):
    """Call build_fn() and retry only on JSONSanityError."""
    from src.json_utils import JSONSanityError

    last_error = None
    for attempt in range(1, max_retries + 2):
        try:
            return build_fn()
        except JSONSanityError as e:
            last_error = e
            if attempt <= max_retries:
                print(
                    f"[run_ollama] {scenario_id} {step_label}: JSON sanity error "
                    f"on attempt {attempt}/{max_retries + 1}: {str(e)[:200]}"
                )
                print(f"[run_ollama]   retrying same scenario...")
            else:
                print(
                    f"[run_ollama] {scenario_id} {step_label}: all "
                    f"{max_retries + 1} attempts failed JSON sanity check"
                )
    raise last_error


def process_scenario(
    scenario: dict,
    output_dir: Path,
    config: dict,
    run_id: str,
) -> dict:
    """
    Run the full pipeline for one scenario via Ollama + local stages.
    Never raises — returns a record with `final_status`.
    """
    scenario_id = scenario.get("id", "?")
    output_dir.mkdir(parents=True, exist_ok=True)

    gen_id = uuid.uuid4().hex
    try:
        _db.create_generation(gen_id, run_id, scenario_id, PLAN_LABEL)
    except Exception as e:
        print(f"[run_ollama] {scenario_id}: DB create_generation failed: {e}")
        return {
            "scenario": scenario,
            "model_label": config.get("model_label", "PuLID + Qwen + Kontext (Ollama, local)"),
            "output_dir": str(output_dir),
            "gen_id": gen_id,
            "run_id": run_id,
            "final_status": "failed",
            "error_stage": "db_create_generation",
            "error_message": f"DB create_generation failed: {e}",
        }

    record: dict = {
        "scenario": scenario,
        "model_label": config.get("model_label", "PuLID + Qwen + Kontext (Ollama, local)"),
        "output_dir": str(output_dir),
        "gen_id": gen_id,
        "run_id": run_id,
        "final_status": "pending",
    }

    # 1. Save scenario yaml
    try:
        (output_dir / "01_scenario.yaml").write_text(
            json.dumps(scenario, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        record["final_status"] = "failed"
        record["error_stage"] = "scenario_save"
        record["error_message"] = f"failed writing 01_scenario.yaml: {e}"
        _db.finalize_generation(gen_id, "failed", record["error_message"])
        return record

    # 2. Step 1 prompt (Ollama) — with JSON-failure retry
    try:
        step_1_output = _call_with_json_retry(
            build_fn=lambda: step_1_prompt_builder_ollama.build_step_1_prompt(scenario),
            scenario_id=scenario_id,
            step_label="Step 1",
        )
    except Exception as e:
        record["final_status"] = "failed"
        record["error_stage"] = "step_1_prompt"
        record["error_message"] = f"Step 1 (Ollama) prompt build failed: {type(e).__name__}: {e}"
        traceback.print_exc()
        _db.update_step_1(gen_id, status="failed", error=record["error_message"])
        _db.finalize_generation(gen_id, "failed", record["error_message"])
        return record

    step_1_text = (step_1_output or {}).get("step_1_image_prompt", "").strip()
    if not step_1_text:
        record["final_status"] = "failed"
        record["error_stage"] = "step_1_prompt"
        record["error_message"] = "step_1_image_prompt empty in Ollama response"
        record["step_1_output"] = step_1_output
        _db.update_step_1(gen_id, status="failed", error=record["error_message"])
        _db.finalize_generation(gen_id, "failed", record["error_message"])
        return record

    (output_dir / "02_step1_prompt.json").write_text(
        json.dumps(step_1_output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    record["step_1_output"] = step_1_output
    _db.update_step_1(gen_id, status="prompt_built", prompt=step_1_text)

    # ─── 3. Stage 1: local FLUX.1-dev + PuLID ──────────────────────────
    pulid_params = step_1_output.get("fal_pulid_params") or config.get(
        "step_1", {}
    ).get("defaults", {})
    persona_out_path = output_dir / "03_step1_persona.jpg"

    step_1_pipeline = None
    try:
        vram_utils.reset_vram_peak()
        step_1_pipeline = step_1_pulid.load_pipeline()
        vram_utils.report_vram("step 1 loaded")

        step_1_meta = step_1_pulid.generate(
            pipeline=step_1_pipeline,
            step_1_prompt=step_1_text,
            fal_pulid_params=pulid_params,
            out_path=persona_out_path,
            scenario_id=scenario_id,
        )
    except Exception as e:
        record["final_status"] = "failed"
        record["error_stage"] = "step_1_pulid"
        record["error_message"] = f"PuLID Stage 1 failed: {type(e).__name__}: {e}"
        traceback.print_exc()
        _db.update_step_1(gen_id, status="failed", error=record["error_message"])
        _db.finalize_generation(gen_id, "failed", record["error_message"])
        return record
    finally:
        if step_1_pipeline is not None:
            vram_utils.unload_pipeline(step_1_pipeline)
            step_1_pipeline = None
            vram_utils.report_vram("step 1 unloaded")

    (output_dir / "03_step1_meta.json").write_text(
        json.dumps(step_1_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    record["step_1_meta"] = step_1_meta
    _db.update_step_1(
        gen_id,
        status="success",
        endpoint=step_1_meta.get("endpoint"),
        image_path=str(persona_out_path),
        request_id=step_1_meta.get("request_id"),
        seed=step_1_meta.get("seed"),
        cost_usd=step_1_meta.get("cost_usd"),
        elapsed_s=step_1_meta.get("elapsed_seconds"),
    )

    # 4. Step 2 prompt (Ollama) — with JSON-failure retry
    try:
        step_2_output = _call_with_json_retry(
            build_fn=lambda: step_2_prompt_builder_ollama.build_step_2_prompt(
                scenario, step_1_output
            ),
            scenario_id=scenario_id,
            step_label="Step 2",
        )
    except Exception as e:
        record["final_status"] = "failed"
        record["error_stage"] = "step_2_prompt"
        record["error_message"] = f"Step 2 (Ollama) prompt build failed: {type(e).__name__}: {e}"
        traceback.print_exc()
        _db.update_step_2(gen_id, status="failed", error=record["error_message"])
        _db.finalize_generation(gen_id, "failed", record["error_message"])
        return record

    step_2_text = (step_2_output or {}).get("step_2_image_prompt", "").strip()
    if not step_2_text:
        record["final_status"] = "failed"
        record["error_stage"] = "step_2_prompt"
        record["error_message"] = "step_2_image_prompt empty in Ollama response"
        record["step_2_output"] = step_2_output
        _db.update_step_2(gen_id, status="failed", error=record["error_message"])
        _db.finalize_generation(gen_id, "failed", record["error_message"])
        return record

    (output_dir / "04_step2_prompt.json").write_text(
        json.dumps(step_2_output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    record["step_2_output"] = step_2_output
    _db.update_step_2(gen_id, status="prompt_built", prompt=step_2_text)

    # ─── 5. Stage 2: local Qwen-Image-Edit-2511 ─────────────────────────
    qwen_params = step_2_output.get("fal_qwen_params") or config.get(
        "step_2", {}
    ).get("defaults", {})
    final_out_path = output_dir / "05_step2_final.jpg"

    step_2_pipeline = None
    try:
        vram_utils.reset_vram_peak()
        step_2_pipeline = step_2_qwen_edit.load_pipeline()
        vram_utils.report_vram("step 2 loaded")

        step_2_meta = step_2_qwen_edit.generate(
            pipeline=step_2_pipeline,
            step_1_local_path=str(persona_out_path),
            step_2_prompt=step_2_text,
            fal_qwen_params=qwen_params,
            out_path=final_out_path,
            scenario_id=scenario_id,
        )
    except Exception as e:
        record["final_status"] = "failed"
        record["error_stage"] = "step_2_qwen"
        record["error_message"] = f"Qwen Stage 2 failed: {type(e).__name__}: {e}"
        traceback.print_exc()
        _db.update_step_2(gen_id, status="failed", error=record["error_message"])
        _db.finalize_generation(gen_id, "failed", record["error_message"])
        return record
    finally:
        if step_2_pipeline is not None:
            vram_utils.unload_pipeline(step_2_pipeline)
            step_2_pipeline = None
            vram_utils.report_vram("step 2 unloaded")

    (output_dir / "05_step2_meta.json").write_text(
        json.dumps(step_2_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    record["step_2_meta"] = step_2_meta
    _db.update_step_2(
        gen_id,
        status="success",
        endpoint=step_2_meta.get("endpoint"),
        image_path=str(final_out_path),
        request_id=step_2_meta.get("request_id"),
        seed=step_2_meta.get("seed"),
        cost_usd=step_2_meta.get("cost_usd"),
        elapsed_s=step_2_meta.get("elapsed_seconds"),
    )

    record["final_status"] = "success"
    record["error_message"] = None
    _db.finalize_generation(gen_id, "success")

    # ─── 6. Stage 3: local FLUX.1-Kontext-dev ───────────────────────────
    # No QC gate in Ollama flow — runs unconditionally after Stage 2 success.
    # Skipped if STEP_3_ENABLED=false. Non-fatal on error.
    step_3_enabled = os.getenv("STEP_3_ENABLED", "true").lower() == "true"

    if step_3_enabled:
        step_3_pipeline = None
        try:
            vram_utils.reset_vram_peak()
            step_3_pipeline = step_3_realism.load_pipeline()
            vram_utils.report_vram("step 3 loaded")

            step_3_out_path = output_dir / "07_step3_realism.jpg"
            lighting_hint = (scenario.get("lighting") or "").strip() or None

            step_3_meta = step_3_realism.generate(
                pipeline=step_3_pipeline,
                step_2_local_path=str(final_out_path),
                out_path=step_3_out_path,
                scenario_id=scenario_id,
                extra_lighting_hint=lighting_hint,
            )
            (output_dir / "07_step3_meta.json").write_text(
                json.dumps(step_3_meta, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            record["step_3_meta"] = step_3_meta
            record["final_image_path"] = str(step_3_out_path)
            print(
                f"[run_ollama] {scenario_id}: Stage 3 realism pass complete → "
                f"07_step3_realism.jpg"
            )
        except Exception as e:
            print(
                f"[run_ollama] {scenario_id}: Stage 3 realism pass failed "
                f"(non-fatal — falling back to Step 2 output): "
                f"{type(e).__name__}: {e}"
            )
            traceback.print_exc()
            record["step_3_meta"] = {"error": str(e)}
            record["final_image_path"] = str(final_out_path)
        finally:
            if step_3_pipeline is not None:
                vram_utils.unload_pipeline(step_3_pipeline)
                step_3_pipeline = None
                vram_utils.report_vram("step 3 unloaded")
    else:
        record["final_image_path"] = str(final_out_path)
        print(f"[run_ollama] {scenario_id}: Stage 3 disabled (STEP_3_ENABLED=false)")

    # 7. chain.html — ollama_flow/outputs/<ts>_<sid>/chain.html → 3 levels
    # up to parent repo root → assets/persona.jpg
    try:
        trace_html.write_chain_html(
            output_dir, record, persona_rel_path="../../../assets/persona.jpg"
        )
    except Exception as e:
        print(f"[run_ollama] {scenario_id}: chain.html write failed (non-fatal): {e}")

    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Alluvi LOCAL image generation pipeline in Ollama mode "
            "(local LLM, $0 cost) for one scenario."
        )
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="scenario id from ../scenarios/scenarios.yaml (e.g. travel_hotel_morning_29)",
    )
    args = parser.parse_args()

    config = _load_config()

    try:
        scenario = scenario_loader.load_scenario(args.scenario)
    except (FileNotFoundError, ValueError) as e:
        print(f"[run_ollama] {e}")
        try:
            all_scenarios = scenario_loader.load_scenarios()
            ids = [s.get("id") for s in all_scenarios]
            preview = ", ".join(ids[:10])
            suffix = "..." if len(ids) > 10 else ""
            print(f"[run_ollama] available ids ({len(ids)}): {preview}{suffix}")
        except Exception:
            pass
        return 1

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{timestamp}_{args.scenario}_ollama"
    output_dir = OUTPUT_ROOT / run_id

    print("")
    print("=" * 72)
    print(f" ALLUVI — OLLAMA SINGLE SCENARIO RUN (local pipeline)")
    print(f" Scenario  : {args.scenario}")
    print(f" Run id    : {run_id}")
    print(f" Output dir: {output_dir}")
    print(f" DB        : {_db.DB_PATH}")
    print("=" * 72)
    print("")

    try:
        _db.create_run(
            run_id=run_id,
            plan=PLAN_LABEL,
            pilot_mode=False,
            notes=f"single scenario (ollama, local): {args.scenario}",
        )
    except Exception as e:
        print(f"[run_ollama] DB create_run failed: {e}")
        return 1

    started = time.time()
    record = process_scenario(scenario, output_dir, config, run_id)
    elapsed = time.time() - started

    final_status = record.get("final_status")
    step_1_meta = record.get("step_1_meta") or {}
    step_2_meta = record.get("step_2_meta") or {}
    step_3_meta = record.get("step_3_meta") or {}

    actual_cost = float(step_1_meta.get("cost_usd") or 0.0)
    actual_cost += float(step_2_meta.get("cost_usd") or 0.0)
    if isinstance(step_3_meta, dict) and not step_3_meta.get("error"):
        actual_cost += float(step_3_meta.get("cost_usd") or 0.0)
    # No Opus/Sonnet cost in Ollama mode

    try:
        _db.finalize_run(
            run_id=run_id,
            total_scenarios=1,
            successful=1 if final_status == "success" else 0,
            failed=0 if final_status == "success" else 1,
            total_cost_usd=actual_cost,
            duration_seconds=int(elapsed),
        )
    except Exception as e:
        print(f"[run_ollama] DB finalize_run failed (non-fatal): {e}")

    print("")
    print("=" * 72)
    print(f" {str(final_status).upper()}")
    print("=" * 72)
    print(f"  wall time:  {elapsed:.1f}s")
    print(f"  chain.html: {output_dir / 'chain.html'}")
    print(f"  run id:     {run_id}  ({_db.DB_PATH})")
    if final_status == "success":
        s1_t = step_1_meta.get("elapsed_seconds", 0)
        s2_t = step_2_meta.get("elapsed_seconds", 0)
        s3_t = step_3_meta.get("elapsed_seconds", 0) if isinstance(step_3_meta, dict) else 0
        print(f"  step 1:     {s1_t:.1f}s (PuLID, local)")
        print(f"  step 2:     {s2_t:.1f}s (Qwen, local)")
        if s3_t:
            print(f"  step 3:     {s3_t:.1f}s (Kontext, local)")
        print(f"  cost:       ${actual_cost:.3f}  (LLM free via Ollama; GPU time at batch level)")
    else:
        print(f"  error stage:   {record.get('error_stage')}")
        print(f"  error message: {record.get('error_message')}")
    print("")

    return 0 if final_status == "success" else 2


if __name__ == "__main__":
    sys.exit(main())