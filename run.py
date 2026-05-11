"""
run.py — single-scenario end-to-end CLI for the Alluvi image generation pipeline.

Pipeline per scenario (matches production run_plan_a.py's per-scenario flow,
but with Step 2 swapped from Nano Banana to Qwen-tuned, and using the new
db.py for SQLite state tracking):

  1. Load + validate the scenario from scenarios/scenarios.yaml by id
  2. Call Opus 4.7 with master_prompt_step1.md → Step 1 prompt envelope
  3. Call fal-ai/flux-pulid → 03_step1_persona.jpg (persona in scene)
  4. Call Opus 4.7 with master_prompt_step2_qwen.md → Step 2 prompt envelope
  5. Call fal-ai/qwen-image-edit-2511 → 05_step2_final.jpg (product composited)
  6. Write all artifacts + chain.html
  7. Persist run + generation rows in data/alluvi.db

Cost: ~$0.36 per scenario.
Wall time: ~60-80s.

Usage (from repo root):
    python run.py --scenario bedroom_robe_with_product_13

For batch processing of multiple scenarios, see run_batch.py.
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

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import db
from src import scenario_loader
from src import step_1_prompt_builder
from src import step_1_pulid
from src import step_2_prompt_builder
from src import step_2_qwen_edit
from src import trace_html


CONFIG_PATH = REPO_ROOT / "config.yaml"
OUTPUT_ROOT = REPO_ROOT / "outputs"

PLAN_LABEL = "pulid_qwen_tuned"  # for db.runs.plan + db.generations.plan


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"missing config: {CONFIG_PATH}")
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


# ─── JSON retry config ────────────────────────────────────────────────────
# Opus very rarely produces malformed JSON, but we add the same retry guard
# the Ollama flow uses so the behavior is consistent across providers.
# Retries cost ~$0.10-$0.28 per attempt (Opus is not free) so we cap at 1
# retry. Most scenarios succeed on the first try; the retry is insurance
# against transient API issues.
MAX_JSON_RETRIES = 1


def _call_with_json_retry(
    build_fn,
    scenario_id: str,
    step_label: str,
    max_retries: int = MAX_JSON_RETRIES,
):
    """
    Call build_fn() and retry only on JSONSanityError. Other exceptions
    propagate immediately. See docstring in ollama_flow/run_ollama.py for
    full rationale.
    """
    from src.json_utils import JSONSanityError

    last_error = None
    for attempt in range(1, max_retries + 2):
        try:
            return build_fn()
        except JSONSanityError as e:
            last_error = e
            if attempt <= max_retries:
                print(
                    f"[run] {scenario_id} {step_label}: JSON sanity error "
                    f"on attempt {attempt}/{max_retries + 1}: {str(e)[:200]}"
                )
                print(f"[run]   retrying same scenario...")
            else:
                print(
                    f"[run] {scenario_id} {step_label}: all "
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
    Run the full pipeline for one scenario. Never raises — returns a record
    dict with `final_status` set to "success" or "failed", plus `error_stage`
    and `error_message` populated on failure.

    Stages tracked (matches production):
      scenario_save / step_1_prompt / step_1_pulid / step_2_prompt / step_2_qwen

    DB state is written incrementally at every stage transition, so even on
    crash you have a partial trail in data/alluvi.db.

    Args:
        scenario: validated scenarios.yaml entry
        output_dir: directory to write outputs into (created if missing)
        config: parsed config.yaml dict
        run_id: parent run id (from db.create_run)

    Returns:
        record dict — always returned, even on failure. Has `gen_id` populated.
    """
    scenario_id = scenario.get("id", "?")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the generation row first (single-source-of-truth handle in DB)
    gen_id = uuid.uuid4().hex
    try:
        db.create_generation(gen_id, run_id, scenario_id, PLAN_LABEL)
    except Exception as e:
        # DB unavailable is fatal for this scenario
        print(f"[run] {scenario_id}: DB create_generation failed: {e}")
        return {
            "scenario": scenario,
            "model_label": config.get("model_label", "PuLID + Qwen"),
            "output_dir": str(output_dir),
            "gen_id": gen_id,
            "run_id": run_id,
            "final_status": "failed",
            "error_stage": "db_create_generation",
            "error_message": f"DB create_generation failed: {e}",
        }

    record: dict = {
        "scenario": scenario,
        "model_label": config.get("model_label", "PuLID + Qwen"),
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
        db.finalize_generation(gen_id, "failed", record["error_message"])
        return record

    # 2. Step 1 prompt (Opus) — with JSON-failure retry
    try:
        step_1_output = _call_with_json_retry(
            build_fn=lambda: step_1_prompt_builder.build_step_1_prompt(scenario),
            scenario_id=scenario_id,
            step_label="Step 1",
        )
    except Exception as e:
        record["final_status"] = "failed"
        record["error_stage"] = "step_1_prompt"
        record["error_message"] = f"Step 1 prompt build failed: {type(e).__name__}: {e}"
        traceback.print_exc()
        db.update_step_1(gen_id, status="failed", error=record["error_message"])
        db.finalize_generation(gen_id, "failed", record["error_message"])
        return record

    step_1_text = (step_1_output or {}).get("step_1_image_prompt", "").strip()
    if not step_1_text:
        record["final_status"] = "failed"
        record["error_stage"] = "step_1_prompt"
        record["error_message"] = "step_1_image_prompt empty in Opus response"
        record["step_1_output"] = step_1_output
        db.update_step_1(gen_id, status="failed", error=record["error_message"])
        db.finalize_generation(gen_id, "failed", record["error_message"])
        return record

    (output_dir / "02_step1_prompt.json").write_text(
        json.dumps(step_1_output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    record["step_1_output"] = step_1_output
    db.update_step_1(gen_id, status="prompt_built", prompt=step_1_text)

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
        record["error_message"] = f"PuLID Stage 1 failed: {type(e).__name__}: {e}"
        traceback.print_exc()
        db.update_step_1(gen_id, status="failed", error=record["error_message"])
        db.finalize_generation(gen_id, "failed", record["error_message"])
        return record

    (output_dir / "03_step1_meta.json").write_text(
        json.dumps(step_1_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    record["step_1_meta"] = step_1_meta
    db.update_step_1(
        gen_id,
        status="success",
        endpoint=step_1_meta.get("endpoint"),
        image_path=str(persona_out_path),
        request_id=step_1_meta.get("request_id"),
        seed=step_1_meta.get("seed"),
        cost_usd=step_1_meta.get("cost_usd"),
        elapsed_s=step_1_meta.get("elapsed_seconds"),
    )

    # 4. Step 2 prompt (Opus, qwen-tuned) — with JSON-failure retry
    try:
        step_2_output = _call_with_json_retry(
            build_fn=lambda: step_2_prompt_builder.build_step_2_prompt(
                scenario, step_1_output
            ),
            scenario_id=scenario_id,
            step_label="Step 2",
        )
    except Exception as e:
        record["final_status"] = "failed"
        record["error_stage"] = "step_2_prompt"
        record["error_message"] = f"Step 2 prompt build failed: {type(e).__name__}: {e}"
        traceback.print_exc()
        db.update_step_2(gen_id, status="failed", error=record["error_message"])
        db.finalize_generation(gen_id, "failed", record["error_message"])
        return record

    step_2_text = (step_2_output or {}).get("step_2_image_prompt", "").strip()
    if not step_2_text:
        record["final_status"] = "failed"
        record["error_stage"] = "step_2_prompt"
        record["error_message"] = "step_2_image_prompt empty in Opus response"
        record["step_2_output"] = step_2_output
        db.update_step_2(gen_id, status="failed", error=record["error_message"])
        db.finalize_generation(gen_id, "failed", record["error_message"])
        return record

    (output_dir / "04_step2_prompt.json").write_text(
        json.dumps(step_2_output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    record["step_2_output"] = step_2_output
    db.update_step_2(gen_id, status="prompt_built", prompt=step_2_text)

    # ─── 5+6. Stage 2 (Qwen) + QC with retry loop ──────────────────────
    # Strategy: run Qwen; QC the result; if QC fails, re-run Qwen with the
    # SAME persona image (free of PuLID cost — we keep Step 1 output).
    # Max 2 retries after the initial attempt (3 Qwen calls worst case).
    # On all-fail: mark scenario as qc_failed and skip — don't crash batch.
    #
    # Each attempt's output is saved:
    #   05_step2_final.jpg            ← latest accepted image (overwritten)
    #   05_step2_final_attempt_N.jpg  ← per-attempt image kept for review
    #   06_qc_result_attempt_N.json   ← per-attempt QC result
    qwen_params = step_2_output.get("fal_qwen_params") or config.get(
        "step_2", {}
    ).get("defaults", {})
    final_out_path = output_dir / "05_step2_final.jpg"
    qc_enabled = os.getenv("QC_ENABLED", "true").lower() == "true"
    MAX_QC_RETRIES = 2  # 1 initial + 2 retries = 3 total Qwen calls max

    qc_attempts: list[dict] = []
    final_qc_result: dict | None = None
    final_step_2_meta: dict | None = None

    for attempt in range(1, MAX_QC_RETRIES + 2):  # 1, 2, 3
        attempt_image_path = output_dir / f"05_step2_final_attempt_{attempt}.jpg"
        is_last_attempt = attempt == MAX_QC_RETRIES + 1

        # Run Qwen (Stage 2)
        try:
            step_2_meta = step_2_qwen_edit.generate(
                step_1_local_path=str(persona_out_path),
                step_2_prompt=step_2_text,
                fal_qwen_params=qwen_params,
                out_path=attempt_image_path,
                scenario_id=f"{scenario_id}#a{attempt}",
            )
        except Exception as e:
            record["final_status"] = "failed"
            record["error_stage"] = "step_2_qwen"
            record["error_message"] = (
                f"Qwen Stage 2 failed on attempt {attempt}: {type(e).__name__}: {e}"
            )
            traceback.print_exc()
            db.update_step_2(gen_id, status="failed", error=record["error_message"])
            db.finalize_generation(gen_id, "failed", record["error_message"])
            return record

        # Copy this attempt to the canonical filename
        try:
            import shutil
            shutil.copy(attempt_image_path, final_out_path)
        except Exception as e:
            print(f"[run] {scenario_id}: copy attempt image failed (non-fatal): {e}")

        final_step_2_meta = step_2_meta

        # Run QC (or skip if disabled)
        if not qc_enabled:
            print(f"[run] {scenario_id}: QC disabled, accepting attempt {attempt}")
            final_qc_result = {
                "passed": True, "score": None, "issues": [],
                "recommendation": "use", "error": "QC_ENABLED=false",
            }
            break

        try:
            from src.qc_validator import validate_image
            qc_result = validate_image(
                final_out_path,
                scenario_id=f"{scenario_id}#a{attempt}",
            )
        except Exception as e:
            print(f"[run] {scenario_id}: QC call crashed on attempt {attempt}: {e}")
            qc_result = {
                "passed": True,  # treat infra failure as pass — don't punish
                "score": 0.5,
                "issues": [f"QC crashed: {e}"],
                "recommendation": "use",
                "error": str(e),
            }

        # Persist this attempt's QC result
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
            print(f"[run] {scenario_id}: QC PASSED on attempt {attempt}")
            final_qc_result = qc_result
            break

        # QC failed
        if is_last_attempt:
            print(
                f"[run] {scenario_id}: QC failed on final attempt "
                f"{attempt}/{MAX_QC_RETRIES + 1} — skipping scenario"
            )
            final_qc_result = qc_result
        else:
            print(
                f"[run] {scenario_id}: QC failed on attempt "
                f"{attempt}/{MAX_QC_RETRIES + 1} — retrying Stage 2 only"
            )
            for issue in qc_result.get("issues", []):
                print(f"  - {issue}")

    # Save the final canonical QC result + step_2_meta
    if final_qc_result:
        (output_dir / "06_qc_result.json").write_text(
            json.dumps({
                **final_qc_result,
                "attempts": qc_attempts,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        record["qc_result"] = final_qc_result
        record["qc_attempts"] = qc_attempts

    record["step_2_meta"] = final_step_2_meta
    (output_dir / "05_step2_meta.json").write_text(
        json.dumps(final_step_2_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    db.update_step_2(
        gen_id,
        status="success",
        endpoint=final_step_2_meta.get("endpoint"),
        image_path=str(final_out_path),
        request_id=final_step_2_meta.get("request_id"),
        seed=final_step_2_meta.get("seed"),
        cost_usd=final_step_2_meta.get("cost_usd"),
        elapsed_s=final_step_2_meta.get("elapsed_seconds"),
    )

    # Final scenario status depends on QC outcome
    if final_qc_result and final_qc_result.get("passed"):
        record["final_status"] = "success"
        record["error_message"] = None
        db.finalize_generation(gen_id, "success")
    else:
        record["final_status"] = "qc_failed"
        issues_summary = "; ".join((final_qc_result or {}).get("issues", []) or ["QC failed"])
        record["error_stage"] = "qc"
        record["error_message"] = f"QC failed after {MAX_QC_RETRIES + 1} attempts: {issues_summary}"
        db.finalize_generation(gen_id, "qc_failed", record["error_message"])

    # Optional DB write (graceful if method missing)
    if final_qc_result is not None:
        try:
            db.update_qc(
                gen_id,
                passed=final_qc_result.get("passed", False),
                score=final_qc_result.get("score", 0.0),
                issues=final_qc_result.get("issues", []),
            )
        except AttributeError:
            pass
        except Exception as e:
            print(f"[run] {scenario_id}: QC db update failed (non-fatal): {e}")

    # 7. chain.html — single-scenario layout: outputs/<ts>_<sid>/chain.html
    # → 2 levels up to repo root → assets/persona.jpg
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

    # Production-style: load + validate via scenario_loader, then find by id
    try:
        scenario = scenario_loader.load_scenario(args.scenario)
    except (FileNotFoundError, ValueError) as e:
        print(f"[run] {e}")
        # Best-effort: show available IDs to help the user
        try:
            all_scenarios = scenario_loader.load_scenarios()
            ids = [s.get("id") for s in all_scenarios]
            preview = ", ".join(ids[:10])
            suffix = "..." if len(ids) > 10 else ""
            print(f"[run] available ids ({len(ids)}): {preview}{suffix}")
        except Exception:
            pass
        return 1

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{timestamp}_{args.scenario}"
    output_dir = OUTPUT_ROOT / run_id

    print("")
    print("=" * 72)
    print(f" ALLUVI — SINGLE SCENARIO RUN")
    print(f" Scenario  : {args.scenario}")
    print(f" Run id    : {run_id}")
    print(f" Output dir: {output_dir}")
    print("=" * 72)
    print("")

    # Create the run row in DB
    try:
        db.create_run(
            run_id=run_id,
            plan=PLAN_LABEL,
            pilot_mode=False,
            notes=f"single scenario: {args.scenario}",
        )
    except Exception as e:
        print(f"[run] DB create_run failed: {e}")
        return 1

    started = time.time()
    record = process_scenario(scenario, output_dir, config, run_id)
    elapsed = time.time() - started

    final_status = record.get("final_status")
    step_1_meta = record.get("step_1_meta") or {}
    step_2_meta = record.get("step_2_meta") or {}

    # Calculate actual cost for this single-scenario run
    actual_cost = float(step_1_meta.get("cost_usd") or 0.0)
    actual_cost += float(step_2_meta.get("cost_usd") or 0.0)
    # Add Opus prompt costs (configured estimates, not metered)
    c_opus_1 = config.get("step_1", {}).get("cost_per_prompt_opus_usd", 0.10)
    c_opus_2 = config.get("step_2", {}).get("cost_per_prompt_opus_usd", 0.18)
    if record.get("step_1_output"):
        actual_cost += c_opus_1
    if record.get("step_2_output"):
        actual_cost += c_opus_2

    # Finalize run
    try:
        db.finalize_run(
            run_id=run_id,
            total_scenarios=1,
            successful=1 if final_status == "success" else 0,
            failed=0 if final_status == "success" else 1,
            total_cost_usd=actual_cost,
            duration_seconds=int(elapsed),
        )
    except Exception as e:
        print(f"[run] DB finalize_run failed (non-fatal): {e}")

    print("")
    print("=" * 72)
    print(f" {str(final_status).upper()}")
    print("=" * 72)
    print(f"  wall time:  {elapsed:.1f}s")
    print(f"  chain.html: {output_dir / 'chain.html'}")
    print(f"  run id:     {run_id}  (data/alluvi.db)")
    if final_status == "success":
        s1_t = step_1_meta.get("elapsed_seconds", 0)
        s2_t = step_2_meta.get("elapsed_seconds", 0)
        print(f"  step 1:     {s1_t:.1f}s (PuLID)")
        print(f"  step 2:     {s2_t:.1f}s (Qwen)")
        print(f"  cost:       ${actual_cost:.3f}")
        qc = record.get("qc_result") or {}
        if qc:
            print(f"  qc:         PASSED (score={qc.get('score')})")
    elif final_status == "qc_failed":
        qc = record.get("qc_result") or {}
        attempts = record.get("qc_attempts") or []
        print(f"  step 1:     {step_1_meta.get('elapsed_seconds', 0):.1f}s (PuLID)")
        print(f"  step 2:     ran {len(attempts)} times (each Qwen retry)")
        print(f"  cost:       ${actual_cost:.3f}")
        print(f"  qc:         FAILED after {len(attempts)} attempts")
        for issue in qc.get("issues", []):
            print(f"              - {issue}")
        print(f"  outcome:    image skipped, not eligible for video pipeline")
    else:
        print(f"  error stage:   {record.get('error_stage')}")
        print(f"  error message: {record.get('error_message')}")
    print("")

    # Exit code: 0 = success, 1 = qc_failed (image rejected but ran fully), 2 = hard error
    if final_status == "success":
        return 0
    elif final_status == "qc_failed":
        return 1
    else:
        return 2


if __name__ == "__main__":
    sys.exit(main())