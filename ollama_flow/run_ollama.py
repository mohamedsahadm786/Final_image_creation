"""
ollama_flow/run_ollama.py — single-scenario runner for Ollama mode.

Replicates the production run.py's per-scenario flow but:
  - Uses Ollama-mode prompt builders from ollama_flow/ollama_src/
  - Routes the SQLite DB to ollama_flow/data/alluvi_ollama.db
    (kept SEPARATE from the production alluvi.db so iteration runs don't
    pollute production evaluation queries)
  - Outputs to ollama_flow/outputs/ (separate from parent's outputs/)
  - Reads ollama_flow/config.yaml (separate from parent's config.yaml)
  - PLAN_LABEL='pulid_qwen_tuned_ollama' so DB rows are distinguishable

Cost: ~$0.08 per scenario (fal API only, no LLM cost).
Wall time: ~80-120s per scenario.

Usage (from inside ollama_flow/):
    python run_ollama.py --scenario bedroom_robe_with_product_13

Why the local Ollama package is named `ollama_src` (not `src`):
    The PARENT repo already has a `src/` package (Final_Image_generation/src/)
    containing db.py, scenario_loader.py, step_1_pulid.py, etc. — which this
    flow REUSES. If we also called our local folder `src/`, Python would hit
    a name collision and shadow one with the other depending on sys.path
    order — fragile. By naming our folder `ollama_src/` we guarantee both
    can coexist on sys.path with no ambiguity.
"""

import argparse
import json
import sys
import time
import traceback
import uuid
import yaml
from datetime import datetime
from pathlib import Path


# Path setup — ollama_flow is at Final_Image_generation/ollama_flow/
OLLAMA_FLOW_ROOT = Path(__file__).resolve().parent
PARENT_REPO_ROOT = OLLAMA_FLOW_ROOT.parent


# Put BOTH on sys.path. Parent goes in LAST so it ends up at sys.path[0]
# and wins any name lookup. (sys.path.insert(0, x) prepends — last insert
# becomes first entry.)
#
# After this:
#   sys.path = [PARENT_REPO_ROOT, OLLAMA_FLOW_ROOT, ...stdlib...]
#
# So `from src import db`         → resolves to PARENT/src/db.py ✓
#    `from ollama_src import X`   → resolves to OLLAMA_FLOW_ROOT/ollama_src/X.py ✓
#
# There's no collision because the two packages have different names.
sys.path.insert(0, str(OLLAMA_FLOW_ROOT))
sys.path.insert(0, str(PARENT_REPO_ROOT))


# ─── Redirect DB to ollama_flow's data/ folder BEFORE any DB calls ──────
# We do this by monkey-patching db.DB_PATH right after import. The parent
# db.py has REPO_ROOT-anchored path; we override it here so this flow's
# runs land in ollama_flow/data/alluvi_ollama.db, NOT the parent's
# data/alluvi.db. Production DB stays untouched.
from src import db as _db
_db.DB_PATH = OLLAMA_FLOW_ROOT / "data" / "alluvi_ollama.db"


# ─── Parent's infrastructure modules (shared with production) ───────────
from src import scenario_loader
from src import step_1_pulid
from src import step_2_qwen_edit
from src import trace_html


# ─── Our own Ollama prompt builders ─────────────────────────────────────
from ollama_src import step_1_prompt_builder_ollama
from ollama_src import step_2_prompt_builder_ollama


# ─── Constants ────────────────────────────────────────────────────────────
CONFIG_PATH = OLLAMA_FLOW_ROOT / "config.yaml"
OUTPUT_ROOT = OLLAMA_FLOW_ROOT / "outputs"

PLAN_LABEL = "pulid_qwen_tuned_ollama"  # tag in DB so we can distinguish from production runs


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"missing config: {CONFIG_PATH}")
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def process_scenario(
    scenario: dict,
    output_dir: Path,
    config: dict,
    run_id: str,
) -> dict:
    """
    Run the full pipeline for one scenario via Ollama. Never raises.
    Same shape as the production process_scenario() but uses Ollama builders.
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
            "model_label": config.get("model_label", "PuLID + Qwen (Ollama)"),
            "output_dir": str(output_dir),
            "gen_id": gen_id,
            "run_id": run_id,
            "final_status": "failed",
            "error_stage": "db_create_generation",
            "error_message": f"DB create_generation failed: {e}",
        }

    record: dict = {
        "scenario": scenario,
        "model_label": config.get("model_label", "PuLID + Qwen (Ollama)"),
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

    # 2. Step 1 prompt (Ollama)
    try:
        step_1_output = step_1_prompt_builder_ollama.build_step_1_prompt(scenario)
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

    # 3. Stage 1: PuLID (reusing parent's caller)
    pulid_params = step_1_output.get("fal_pulid_params") or config.get(
        "step_1", {}
    ).get("defaults", {})
    persona_out_path = output_dir / "03_step1_persona.jpg"

    try:
        step_1_meta = step_1_pulid.generate(
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

    # 4. Step 2 prompt (Ollama)
    try:
        step_2_output = step_2_prompt_builder_ollama.build_step_2_prompt(
            scenario, step_1_output
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

    # 5. Stage 2: Qwen (reusing parent's caller)
    qwen_params = step_2_output.get("fal_qwen_params") or config.get(
        "step_2", {}
    ).get("defaults", {})
    final_out_path = output_dir / "05_step2_final.jpg"

    try:
        step_2_meta = step_2_qwen_edit.generate(
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

    # 6. chain.html
    # Path math: ollama_flow/outputs/<ts>_<sid>/chain.html
    # → 3 levels up to parent repo root → assets/persona.jpg
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
            "Run the Alluvi image generation pipeline in Ollama mode "
            "(local LLM, no Anthropic cost) for one scenario."
        )
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="scenario id from ../scenarios/scenarios.yaml (e.g. bedroom_robe_with_product_13)",
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
    print(f" ALLUVI — OLLAMA SINGLE SCENARIO RUN")
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
            notes=f"single scenario (ollama): {args.scenario}",
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

    actual_cost = float(step_1_meta.get("cost_usd") or 0.0)
    actual_cost += float(step_2_meta.get("cost_usd") or 0.0)
    # No Opus cost in Ollama mode

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
        print(f"  step 1:     {s1_t:.1f}s (PuLID)")
        print(f"  step 2:     {s2_t:.1f}s (Qwen)")
        print(f"  cost:       ${actual_cost:.3f}  (fal only, LLM was free via Ollama)")
    else:
        print(f"  error stage:   {record.get('error_stage')}")
        print(f"  error message: {record.get('error_message')}")
    print("")

    return 0 if final_status == "success" else 2


if __name__ == "__main__":
    sys.exit(main())