"""
run.py — single-scenario end-to-end CLI for the Alluvi image generation pipeline.

Pipeline for one scenario:
  1. Load the scenario record from scenarios/scenarios.yaml by id
  2. Call Opus 4.7 with master_prompt_step1.md → Step 1 prompt envelope
  3. Call fal-ai/flux-pulid → 03_step1_persona.jpg (persona in scene)
  4. Call Opus 4.7 with master_prompt_step2_qwen.md → Step 2 prompt envelope
  5. Call fal-ai/qwen-image-edit-2511 → 05_step2_final.jpg (product composited)
  6. Write all artifacts + chain.html

Cost: ~$0.42 per scenario.
Wall time: ~60-80s.

Usage (from repo root):
    python run.py --scenario bedroom_robe_with_product_13

For batch processing of multiple scenarios, see run_batch.py.
"""

import argparse
import json
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import step_1_prompt_builder
from src import step_1_pulid
from src import step_2_prompt_builder
from src import step_2_qwen_edit
from src import trace_html


CONFIG_PATH = REPO_ROOT / "config.yaml"
SCENARIOS_PATH = REPO_ROOT / "scenarios" / "scenarios.yaml"
OUTPUT_ROOT = REPO_ROOT / "outputs"


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"missing config: {CONFIG_PATH}")
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _load_all_scenarios() -> list[dict]:
    if not SCENARIOS_PATH.exists():
        raise FileNotFoundError(f"missing scenarios file: {SCENARIOS_PATH}")
    data = yaml.safe_load(SCENARIOS_PATH.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        scenarios = data.get("scenarios", []) or data.get("items", []) or []
    elif isinstance(data, list):
        scenarios = data
    else:
        scenarios = []
    if not scenarios:
        raise RuntimeError(f"no scenarios found in {SCENARIOS_PATH}")
    return [s for s in scenarios if isinstance(s, dict) and s.get("id")]


def _find_scenario(scenario_id: str) -> dict | None:
    """Find a scenario by id. Returns None if not found."""
    all_scenarios = _load_all_scenarios()
    for s in all_scenarios:
        if s.get("id") == scenario_id:
            return s
    return None


def process_scenario(scenario: dict, output_dir: Path, config: dict) -> dict:
    """
    Run the full pipeline for one scenario. Never raises — returns a record
    dict with `final_status` set to "success" or "failed", plus `error_stage`
    and `error_message` populated on failure.

    Args:
        scenario: parsed scenarios.yaml entry
        output_dir: directory to write outputs into (created if missing)
        config: parsed config.yaml dict

    Returns:
        record dict — always returned, even on failure.
    """
    scenario_id = scenario.get("id", "?")
    output_dir.mkdir(parents=True, exist_ok=True)

    record: dict = {
        "scenario": scenario,
        "model_label": config.get("model_label", "PuLID + Qwen"),
        "output_dir": str(output_dir),
        "final_status": "pending",
    }

    # 1. Save scenario yaml
    try:
        (output_dir / "01_scenario.yaml").write_text(
            yaml.safe_dump(scenario, sort_keys=False), encoding="utf-8"
        )
    except Exception as e:
        record["final_status"] = "failed"
        record["error_stage"] = "scenario_save"
        record["error_message"] = f"failed writing 01_scenario.yaml: {e}"
        return record

    # 2. Step 1 prompt (Opus)
    try:
        step_1_output = step_1_prompt_builder.build_step_1_prompt(scenario)
    except Exception as e:
        record["final_status"] = "failed"
        record["error_stage"] = "step_1_prompt"
        record["error_message"] = f"Step 1 prompt build failed: {e}"
        return record

    step_1_text = (step_1_output or {}).get("step_1_image_prompt", "").strip()
    if not step_1_text:
        record["final_status"] = "failed"
        record["error_stage"] = "step_1_prompt"
        record["error_message"] = "step_1_image_prompt empty in Opus response"
        record["step_1_output"] = step_1_output
        return record

    (output_dir / "02_step1_prompt.json").write_text(
        json.dumps(step_1_output, indent=2), encoding="utf-8"
    )
    record["step_1_output"] = step_1_output

    # 3. Stage 1: PuLID
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
        record["error_message"] = f"PuLID Stage 1 failed: {e}"
        return record

    (output_dir / "03_step1_meta.json").write_text(
        json.dumps(step_1_meta, indent=2), encoding="utf-8"
    )
    record["step_1_meta"] = step_1_meta

    # 4. Step 2 prompt (Opus, qwen-tuned)
    try:
        step_2_output = step_2_prompt_builder.build_step_2_prompt(
            scenario, step_1_output
        )
    except Exception as e:
        record["final_status"] = "failed"
        record["error_stage"] = "step_2_prompt"
        record["error_message"] = f"Step 2 prompt build failed: {e}"
        return record

    step_2_text = (step_2_output or {}).get("step_2_image_prompt", "").strip()
    if not step_2_text:
        record["final_status"] = "failed"
        record["error_stage"] = "step_2_prompt"
        record["error_message"] = "step_2_image_prompt empty in Opus response"
        record["step_2_output"] = step_2_output
        return record

    (output_dir / "04_step2_prompt.json").write_text(
        json.dumps(step_2_output, indent=2), encoding="utf-8"
    )
    record["step_2_output"] = step_2_output

    # 5. Stage 2: Qwen
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
        record["error_message"] = f"Qwen Stage 2 failed: {e}"
        return record

    (output_dir / "05_step2_meta.json").write_text(
        json.dumps(step_2_meta, indent=2), encoding="utf-8"
    )
    record["step_2_meta"] = step_2_meta

    record["final_status"] = "success"
    record["error_message"] = None

    # 6. chain.html
    # Single-scenario layout: outputs/<ts>_<sid>/chain.html → 2 levels up to repo
    try:
        trace_html.write_chain_html(
            output_dir, record, persona_rel_path="../../assets/persona.jpg"
        )
    except Exception as e:
        print(f"[run] {scenario_id}: chain.html write failed (non-fatal): {e}")

    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full Alluvi image generation pipeline (PuLID + Qwen) "
            "for one scenario."
        )
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="scenario id from scenarios.yaml (e.g. bedroom_robe_with_product_13)",
    )
    args = parser.parse_args()

    config = _load_config()

    scenario = _find_scenario(args.scenario)
    if scenario is None:
        print(f"[run] scenario '{args.scenario}' not found in {SCENARIOS_PATH}")
        all_ids = [s.get("id") for s in _load_all_scenarios()]
        print(f"[run] available ids ({len(all_ids)}): {', '.join(all_ids[:10])}...")
        return 1

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = OUTPUT_ROOT / f"{timestamp}_{args.scenario}"

    print(f"[run] output dir: {output_dir}")
    print(f"[run] scenario:   {args.scenario}")
    print("")

    started = time.time()
    record = process_scenario(scenario, output_dir, config)
    elapsed = time.time() - started

    final_status = record.get("final_status")
    step_1_meta = record.get("step_1_meta") or {}
    step_2_meta = record.get("step_2_meta") or {}

    print("")
    print("[run] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"[run] {str(final_status).upper()}")
    print(f"[run]   wall time:  {elapsed:.1f}s")
    print(f"[run]   chain.html: {output_dir / 'chain.html'}")
    if final_status == "success":
        s1_t = step_1_meta.get("elapsed_seconds", 0)
        s2_t = step_2_meta.get("elapsed_seconds", 0)
        print(f"[run]   step 1:     {s1_t:.1f}s (PuLID)")
        print(f"[run]   step 2:     {s2_t:.1f}s (Qwen)")
        cost_per = config.get("cost_per_scenario_usd", 0.42)
        print(f"[run]   est. cost:  ${cost_per:.3f}")
    else:
        print(f"[run]   error stage:   {record.get('error_stage')}")
        print(f"[run]   error message: {record.get('error_message')}")
    print("[run] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    return 0 if final_status == "success" else 2


if __name__ == "__main__":
    sys.exit(main())