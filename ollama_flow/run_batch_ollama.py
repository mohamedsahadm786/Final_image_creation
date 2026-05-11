"""
ollama_flow/run_batch_ollama.py — batch runner for Ollama mode.

Same orchestration shape as the parent's run_batch.py but uses Ollama prompt
builders and the separated ollama_flow DB / outputs / config.

Per-scenario cost: ~$0.08 (fal only, no LLM cost)
30-scenario batch cost: ~$2.40
Wall time (30 sequential): ~50-60 min (Ollama adds 5-15s/call vs Claude)

Usage (from inside ollama_flow/):
    python run_batch_ollama.py
    python run_batch_ollama.py --pilot                # 5-scenario validation
    python run_batch_ollama.py --only ID1,ID2         # specific scenarios only
    python run_batch_ollama.py --exclude ID3          # skip specific scenarios
    python run_batch_ollama.py --yes                  # skip cost confirmation
    python run_batch_ollama.py --skip-preflight       # not recommended
"""

import argparse
import json
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path

OLLAMA_FLOW_ROOT = Path(__file__).resolve().parent
PARENT_REPO_ROOT = OLLAMA_FLOW_ROOT.parent

# Parent goes in LAST so it ends up at sys.path[0] and wins name lookups.
# (sys.path.insert(0, x) prepends — last insert becomes first entry.)
sys.path.insert(0, str(OLLAMA_FLOW_ROOT))
sys.path.insert(0, str(PARENT_REPO_ROOT))


# Import process_scenario and the DB-redirected `_db` from run_ollama.py.
# This is the single source of truth for the Ollama per-scenario pipeline.
from run_ollama import (
    process_scenario,
    _load_config,
    _db,
    PLAN_LABEL,
    OUTPUT_ROOT,
)

# Parent modules we reuse
from src import scenario_loader
from src import trace_html
from src import overview_html

# Our own preflight
from preflight_ollama import run_preflight


CONFIG_PATH = OLLAMA_FLOW_ROOT / "config.yaml"


def _filter_scenarios(
    scenarios: list[dict],
    only: list[str] | None,
    exclude: list[str] | None,
) -> list[dict]:
    if only:
        only_set = set(only)
        scenarios = [s for s in scenarios if s.get("id") in only_set]
    if exclude:
        excl_set = set(exclude)
        scenarios = [s for s in scenarios if s.get("id") not in excl_set]
    return scenarios


def _confirm_cost(n: int, config: dict, yes: bool) -> bool:
    cost_per = config.get("cost_per_scenario_usd", 0.08)
    total = cost_per * n
    # Ollama latency adds ~15s vs Claude per scenario; estimate ~100s/scenario
    est_seconds = n * 100

    print("")
    print(f"[batch_ollama] scenarios:       {n}")
    print(f"[batch_ollama] per scenario:    ${cost_per:.3f}  (fal only, LLM is free)")
    print(f"[batch_ollama] estimated cost:  ${total:.2f}")
    print(
        f"[batch_ollama] est. wall time:  ~{est_seconds // 60} min "
        f"({est_seconds}s sequential)"
    )

    if yes:
        print("[batch_ollama] --yes flag set, skipping confirmation")
        return True

    answer = input("[batch_ollama] proceed? (y/n): ").strip().lower()
    return answer in ("y", "yes")


def _build_summary(
    batch_dir: Path,
    records: list[dict],
    started_at: float,
    interrupted: bool,
    config: dict,
) -> dict:
    succeeded = sum(1 for r in records if r.get("final_status") == "success")
    failed = len(records) - succeeded

    c_pulid = config.get("step_1", {}).get("cost_per_image_usd", 0.04)
    c_qwen = config.get("step_2", {}).get("cost_per_image_usd", 0.04)
    c_total = config.get("cost_per_scenario_usd", c_pulid + c_qwen)

    actual_cost = 0.0
    for r in records:
        stage = r.get("error_stage")
        if r.get("final_status") == "success":
            actual_cost += c_total
        else:
            # Ollama LLM calls are free, so only fal contributes to cost
            if stage in ("scenario_save", "db_create_generation", "step_1_prompt"):
                actual_cost += 0.0
            elif stage == "step_1_pulid":
                actual_cost += c_pulid
            elif stage == "step_2_prompt":
                actual_cost += c_pulid
            elif stage == "step_2_qwen":
                actual_cost += c_pulid + (c_qwen * 0.5)
            else:
                actual_cost += 0.0

    return {
        "succeeded": succeeded,
        "failed": failed,
        "actual_cost_usd": round(actual_cost, 3),
        "elapsed_seconds": time.time() - started_at,
        "timestamp": batch_dir.name,
        "model_label": config.get("model_label", "PuLID + Qwen (Ollama)"),
        "interrupted": interrupted,
    }


def _write_overview_and_manifest(
    batch_dir: Path,
    records: list[dict],
    started_at: float,
    interrupted: bool,
    config: dict,
    run_id: str,
) -> None:
    summary = _build_summary(batch_dir, records, started_at, interrupted, config)

    overview_html.write_overview_html(batch_dir, records, summary)

    manifest = {
        **summary,
        "run_id": run_id,
        "db_path": str(_db.DB_PATH),
        "llm_provider": "ollama",
        "scenario_records": [
            {
                "scenario_id": r.get("scenario", {}).get("id", "?"),
                "gen_id": r.get("gen_id"),
                "final_status": r.get("final_status"),
                "error_stage": r.get("error_stage"),
                "error_message": r.get("error_message"),
                "step_1_word_count": r.get("step_1_output", {}).get("word_count"),
                "step_2_word_count": r.get("step_2_output", {}).get("word_count"),
                "step_1_seed": (r.get("step_1_meta") or {}).get("seed"),
                "step_2_seed": (r.get("step_2_meta") or {}).get("seed"),
                "step_1_elapsed_s": (r.get("step_1_meta") or {}).get("elapsed_seconds"),
                "step_2_elapsed_s": (r.get("step_2_meta") or {}).get("elapsed_seconds"),
                "id_weight": (
                    r.get("step_1_output", {}).get("fal_pulid_params") or {}
                ).get("id_weight"),
                "true_cfg": (
                    r.get("step_1_output", {}).get("fal_pulid_params") or {}
                ).get("true_cfg"),
            }
            for r in records
        ],
    }
    (batch_dir / "batch_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Batch runner for the Alluvi image generation pipeline in OLLAMA MODE "
            "(local LLM, no Anthropic cost). Outputs and DB are kept SEPARATE "
            "from the production Claude flow."
        )
    )
    parser.add_argument("--pilot", action="store_true",
                        help="run only the 5 pilot scenarios")
    parser.add_argument("--only", type=str, default=None,
                        help="comma-separated scenario IDs to include")
    parser.add_argument("--exclude", type=str, default=None,
                        help="comma-separated scenario IDs to skip")
    parser.add_argument("--yes", action="store_true",
                        help="skip cost confirmation prompt")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="skip the preflight check (not recommended)")
    args = parser.parse_args()

    config = _load_config()

    if not args.skip_preflight:
        errors, _warnings = run_preflight(verbose=True)
        if errors:
            print("[batch_ollama] preflight failed — aborting before any API call")
            return 1
    else:
        print("[batch_ollama] --skip-preflight set — skipping preflight check")

    try:
        all_scenarios = scenario_loader.load_scenarios()
    except (FileNotFoundError, ValueError) as e:
        print(f"[batch_ollama] scenario loading failed:")
        print(f"[batch_ollama]   {e}")
        return 1

    if args.pilot:
        scenarios = scenario_loader.filter_pilot_scenarios(all_scenarios)
        mode_label = f"pilot ({len(scenarios)})"
    else:
        scenarios = list(all_scenarios)
        mode_label = f"full ({len(scenarios)})"

    only = [s.strip() for s in args.only.split(",")] if args.only else None
    exclude = [s.strip() for s in args.exclude.split(",")] if args.exclude else None
    scenarios = _filter_scenarios(scenarios, only, exclude)

    if not scenarios:
        print(f"[batch_ollama] no scenarios to run after filtering")
        return 1

    if not _confirm_cost(len(scenarios), config, args.yes):
        print("[batch_ollama] aborted by user")
        return 0

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{timestamp}_batch_ollama"
    batch_dir = OUTPUT_ROOT / run_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    print("")
    print("=" * 72)
    print(f" ALLUVI OLLAMA BATCH RUN: {run_id}")
    print(f" Mode: {mode_label}")
    print(f" Output dir: {batch_dir}")
    print(f" Scenarios queued: {len(scenarios)}")
    print(f" DB: {_db.DB_PATH}")
    print("=" * 72)
    print("")

    try:
        _db.create_run(
            run_id=run_id,
            plan=PLAN_LABEL,
            pilot_mode=args.pilot,
            notes=f"ollama batch — mode={mode_label}",
        )
    except Exception as e:
        print(f"[batch_ollama] DB create_run failed: {e}")
        return 1

    records: list[dict] = []
    started_at = time.time()
    interrupted = False

    try:
        for i, scenario in enumerate(scenarios, start=1):
            sc_id = scenario.get("id", f"unknown_{i}")
            scenario_started = time.time()
            print(f"\n{'─' * 72}")
            print(
                f" [{i}/{len(scenarios)}] {sc_id} — starting "
                f"({i-1} done, {len(scenarios)-i+1} remaining)"
            )
            print(f"{'─' * 72}")

            output_dir = batch_dir / sc_id

            try:
                record = process_scenario(scenario, output_dir, config, run_id)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"[batch_ollama]   {sc_id}: UNEXPECTED CRASH: {e}")
                record = {
                    "scenario": scenario,
                    "final_status": "failed",
                    "error_stage": "unknown",
                    "error_message": f"unhandled exception: {e}",
                    "model_label": config.get("model_label", "PuLID + Qwen (Ollama)"),
                    "run_id": run_id,
                }

            # Batch mode chain.html: ollama_flow/outputs/<batch>/<scenario>/chain.html
            # → 4 levels up to repo parent → assets/persona.jpg
            try:
                trace_html.write_chain_html(
                    output_dir,
                    record,
                    persona_rel_path="../../../../assets/persona.jpg",
                )
            except Exception as e:
                print(f"[batch_ollama]   {sc_id}: chain.html write failed (non-fatal): {e}")

            records.append(record)

            elapsed = time.time() - scenario_started
            status = record.get("final_status", "?")
            stage_note = (
                f" (failed at: {record.get('error_stage', '?')})"
                if status != "success" else ""
            )
            check = "✓" if status == "success" else "✗"
            print(f"  {check} {sc_id}: {status.upper()} in {elapsed:.1f}s{stage_note}")

            if i % 5 == 0:
                _write_overview_and_manifest(
                    batch_dir, records, started_at, interrupted=False,
                    config=config, run_id=run_id,
                )
                print(f"  (overview.html refreshed at {i}/{len(scenarios)})")

    except KeyboardInterrupt:
        interrupted = True
        print("")
        print("=" * 72)
        print(" interrupted by user (Ctrl+C)")
        print(f" writing partial overview for {len(records)} completed scenarios...")
        print("=" * 72)

    _write_overview_and_manifest(
        batch_dir, records, started_at, interrupted=interrupted,
        config=config, run_id=run_id,
    )

    succeeded = sum(1 for r in records if r.get("final_status") == "success")
    failed = len(records) - succeeded
    elapsed_total = time.time() - started_at

    summary = _build_summary(batch_dir, records, started_at, interrupted, config)
    actual_cost = summary["actual_cost_usd"]

    try:
        _db.finalize_run(
            run_id=run_id,
            total_scenarios=len(records),
            successful=succeeded,
            failed=failed,
            total_cost_usd=actual_cost,
            duration_seconds=int(elapsed_total),
        )
    except Exception as e:
        print(f"[batch_ollama] DB finalize_run failed (non-fatal): {e}")

    failure_stages: dict[str, int] = {}
    for r in records:
        if r.get("final_status") != "success":
            stage = r.get("error_stage", "unknown")
            failure_stages[stage] = failure_stages.get(stage, 0) + 1

    print("")
    print("=" * 72)
    print(f" OLLAMA BATCH COMPLETE{'  (interrupted)' if interrupted else ''}")
    print("=" * 72)
    print(f"  run id:        {run_id}")
    print(f"  succeeded:     {succeeded} / {len(records)}")
    print(f"  failed:        {failed}")
    print(f"  actual cost:   ${actual_cost:.2f}  (fal only, LLM was free)")
    print(f"  wall time:     {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    if failure_stages:
        print(f"  failures by stage:  "
              + ", ".join(f"{k}={v}" for k, v in sorted(failure_stages.items())))
    print(f"  overview.html: {batch_dir / 'overview.html'}")
    print(f"  manifest.json: {batch_dir / 'batch_manifest.json'}")
    print(f"  DB:            {_db.DB_PATH}  (run_id={run_id})")
    print("")

    return 0 if (failed == 0 and not interrupted) else 2


if __name__ == "__main__":
    sys.exit(main())