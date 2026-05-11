"""
run_batch.py — multi-scenario batch CLI for the Alluvi image generation pipeline.

Matches production run_plan_a.py's orchestration shape, but:
  - Step 2 is Qwen-tuned (not Nano Banana)
  - Uses src/db.py (SQLite) for state tracking — same schema as production
  - Auto-invokes preflight before any API call (replaces old _verify_inputs)
  - Adds pilot/only/exclude filters + cost confirmation + mid-batch HTML refresh
  - Ctrl+C-safe partial outputs

Per-scenario cost: ~$0.36
30-scenario batch cost: ~$10.80
Wall time (30 sequential): ~35-40 min

Outputs persist in TWO places:
  - outputs/<ts>_batch/<scenario>/ — filesystem (chain.html, JPGs, JSONs)
  - data/alluvi.db                  — SQLite (runs + generations tables)

Usage (from repo root):
    python run_batch.py
    python run_batch.py --pilot                # 5-scenario validation run
    python run_batch.py --only ID1,ID2         # specific scenarios only
    python run_batch.py --exclude ID3,ID4      # skip specific scenarios
    python run_batch.py --yes                  # skip cost confirmation
    python run_batch.py --skip-preflight       # skip preflight (not recommended)
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

# Reuse process_scenario from run.py (single source of truth for the per-
# scenario pipeline). PLAN_LABEL is shared so DB rows are consistent.
from run import process_scenario, _load_config, PLAN_LABEL

from preflight import run_preflight
from src import db
from src import scenario_loader
from src import trace_html
from src import overview_html


CONFIG_PATH = REPO_ROOT / "config.yaml"
OUTPUT_ROOT = REPO_ROOT / "outputs"


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
    cost_per = config.get("cost_per_scenario_usd", 0.36)
    total = cost_per * n
    # ~70s/scenario sequential: PuLID ~25s + Opus x2 ~15s + Qwen ~30s
    est_seconds = n * 70

    print("")
    print(f"[batch] scenarios:       {n}")
    print(f"[batch] per scenario:    ${cost_per:.3f}")
    print(f"[batch] estimated cost:  ${total:.2f}")
    print(
        f"[batch] est. wall time:  ~{est_seconds // 60} min "
        f"({est_seconds}s sequential)"
    )

    if yes:
        print("[batch] --yes flag set, skipping confirmation")
        return True

    answer = input("[batch] proceed? (y/n): ").strip().lower()
    return answer in ("y", "yes")


def _build_summary(
    batch_dir: Path,
    records: list[dict],
    started_at: float,
    interrupted: bool,
    config: dict,
) -> dict:
    """Build the summary dict + estimate actual cost given per-stage failures."""
    succeeded = sum(1 for r in records if r.get("final_status") == "success")
    failed = len(records) - succeeded

    # Per-stage costs for partial-failure accounting
    c_opus_1 = config.get("step_1", {}).get("cost_per_prompt_opus_usd", 0.10)
    c_pulid = config.get("step_1", {}).get("cost_per_image_usd", 0.04)
    c_opus_2 = config.get("step_2", {}).get("cost_per_prompt_opus_usd", 0.18)
    c_qwen = config.get("step_2", {}).get("cost_per_image_usd", 0.04)
    c_total = config.get(
        "cost_per_scenario_usd", c_opus_1 + c_pulid + c_opus_2 + c_qwen
    )

    actual_cost = 0.0
    for r in records:
        stage = r.get("error_stage")
        if r.get("final_status") == "success":
            actual_cost += c_total
        else:
            if stage in ("scenario_save", "db_create_generation"):
                actual_cost += 0.0
            elif stage == "step_1_prompt":
                actual_cost += c_opus_1
            elif stage == "step_1_pulid":
                actual_cost += c_opus_1 + c_pulid
            elif stage == "step_2_prompt":
                actual_cost += c_opus_1 + c_pulid + c_opus_2
            elif stage == "step_2_qwen":
                # Step 2 partially executed — charge half of Qwen cost
                actual_cost += c_opus_1 + c_pulid + c_opus_2 + (c_qwen * 0.5)
            else:
                actual_cost += c_opus_1

    return {
        "succeeded": succeeded,
        "failed": failed,
        "actual_cost_usd": round(actual_cost, 3),
        "elapsed_seconds": time.time() - started_at,
        "timestamp": batch_dir.name,
        "model_label": config.get("model_label", "PuLID + Qwen"),
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
        "db_path": str(REPO_ROOT / "data" / "alluvi.db"),
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
            "Batch runner for the Alluvi image generation pipeline "
            "(PuLID Stage 1 + Qwen Stage 2)."
        )
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help=(
            "run only the 5 pilot scenarios (3 hero flat-lays + 2 easy non-hero). "
            "Cheapest way to validate the pipeline before a full run."
        ),
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="comma-separated scenario IDs to include (default: all)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="comma-separated scenario IDs to skip",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip cost confirmation prompt",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="skip the preflight check (not recommended)",
    )
    args = parser.parse_args()

    config = _load_config()

    # Preflight FIRST — fails fast on missing files, missing env vars,
    # scenarios.yaml schema errors, missing deps, missing modules.
    # Replaces the previous _verify_inputs() which was a subset of this.
    if not args.skip_preflight:
        errors, _warnings = run_preflight(verbose=True)
        if errors:
            print("[batch] preflight failed — aborting before any API call")
            return 1
    else:
        print("[batch] --skip-preflight set — skipping preflight check")

    # Apply pilot / only / exclude filters
    try:
        all_scenarios = scenario_loader.load_scenarios()
    except (FileNotFoundError, ValueError) as e:
        # Should be caught by preflight, but defense in depth
        print(f"[batch] scenario loading failed:")
        print(f"[batch]   {e}")
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
        print(
            f"[batch] no scenarios to run after filtering "
            f"({len(all_scenarios)} available)"
        )
        if only:
            print(f"[batch]   --only filter: {only}")
        if exclude:
            print(f"[batch]   --exclude filter: {exclude}")
        if args.pilot:
            print(f"[batch]   --pilot filter active")
        return 1

    if not _confirm_cost(len(scenarios), config, args.yes):
        print("[batch] aborted by user")
        return 0

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{timestamp}_batch"
    batch_dir = OUTPUT_ROOT / run_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    print("")
    print("=" * 72)
    print(f" ALLUVI BATCH RUN: {run_id}")
    print(f" Mode: {mode_label}")
    print(f" Output dir: {batch_dir}")
    print(f" Scenarios queued: {len(scenarios)}")
    print(f" DB: {REPO_ROOT / 'data' / 'alluvi.db'}")
    print("=" * 72)
    print("")

    # Create run row in DB BEFORE any scenario processing
    try:
        db.create_run(
            run_id=run_id,
            plan=PLAN_LABEL,
            pilot_mode=args.pilot,
            notes=f"mode={mode_label}",
        )
    except Exception as e:
        print(f"[batch] DB create_run failed: {e}")
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
                print(f"[batch]   {sc_id}: UNEXPECTED CRASH: {e}")
                record = {
                    "scenario": scenario,
                    "final_status": "failed",
                    "error_stage": "unknown",
                    "error_message": f"unhandled exception: {e}",
                    "model_label": config.get("model_label", "PuLID + Qwen"),
                    "run_id": run_id,
                }

            # Batch mode chain.html uses a different relative path to persona.jpg
            # (3 levels up instead of 2).
            try:
                trace_html.write_chain_html(
                    output_dir,
                    record,
                    persona_rel_path="../../../assets/persona.jpg",
                )
            except Exception as e:
                print(
                    f"[batch]   {sc_id}: chain.html write failed (non-fatal): {e}"
                )

            records.append(record)

            elapsed = time.time() - scenario_started
            status = record.get("final_status", "?")
            stage_note = (
                f" (failed at: {record.get('error_stage', '?')})"
                if status != "success"
                else ""
            )
            check = "✓" if status == "success" else "✗"
            print(
                f"  {check} {sc_id}: {status.upper()} in {elapsed:.1f}s{stage_note}"
            )

            # Mid-batch refresh every 5 scenarios
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
        print(
            f" writing partial overview for {len(records)} completed scenarios..."
        )
        print("=" * 72)

    _write_overview_and_manifest(
        batch_dir, records, started_at, interrupted=interrupted,
        config=config, run_id=run_id,
    )

    succeeded = sum(1 for r in records if r.get("final_status") == "success")
    failed = len(records) - succeeded
    elapsed_total = time.time() - started_at

    # Calculate total actual cost for finalize_run
    summary = _build_summary(batch_dir, records, started_at, interrupted, config)
    actual_cost = summary["actual_cost_usd"]

    # Finalize run row in DB
    try:
        db.finalize_run(
            run_id=run_id,
            total_scenarios=len(records),
            successful=succeeded,
            failed=failed,
            total_cost_usd=actual_cost,
            duration_seconds=int(elapsed_total),
        )
    except Exception as e:
        print(f"[batch] DB finalize_run failed (non-fatal): {e}")

    failure_stages: dict[str, int] = {}
    for r in records:
        if r.get("final_status") != "success":
            stage = r.get("error_stage", "unknown")
            failure_stages[stage] = failure_stages.get(stage, 0) + 1

    print("")
    print("=" * 72)
    print(f" BATCH COMPLETE{'  (interrupted)' if interrupted else ''}")
    print("=" * 72)
    print(f"  run id:        {run_id}")
    print(f"  succeeded:     {succeeded} / {len(records)}")
    print(f"  failed:        {failed}")
    print(f"  actual cost:   ${actual_cost:.2f}")
    print(
        f"  wall time:     {elapsed_total:.0f}s "
        f"({elapsed_total/60:.1f} min)"
    )
    if failure_stages:
        print(
            f"  failures by stage:  "
            + ", ".join(f"{k}={v}" for k, v in sorted(failure_stages.items()))
        )
    print(f"  overview.html: {batch_dir / 'overview.html'}")
    print(f"  manifest.json: {batch_dir / 'batch_manifest.json'}")
    print(f"  DB:            {REPO_ROOT / 'data' / 'alluvi.db'}  (run_id={run_id})")
    print("")

    return 0 if (failed == 0 and not interrupted) else 2


if __name__ == "__main__":
    sys.exit(main())