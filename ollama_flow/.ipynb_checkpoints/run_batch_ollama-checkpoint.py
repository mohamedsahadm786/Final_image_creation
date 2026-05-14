"""
ollama_flow/run_batch_ollama.py — batch runner for Ollama mode (LOCAL stages).

Mirrors orchestration/per_scenario/run_batch.py but:
  - Calls process_scenario() from run_ollama.py (Ollama prompt builders)
  - Routes DB writes to ollama_flow/data/alluvi_ollama.db
  - Outputs to ollama_flow/outputs/<ts>_batch[_pilot]/<scenario_id>/

Per-scenario load pattern: each scenario loads → infers → unloads each
stage in turn. 3 model loads per scenario. For stage-batched optimization
(3 loads total), see future orchestration/stage_batched/ — out of scope here.

CLI:
    python run_batch_ollama.py                    # all scenarios
    python run_batch_ollama.py --pilot            # first 5
    python run_batch_ollama.py --only ID1,ID2     # specific IDs
    python run_batch_ollama.py --exclude ID1,ID2  # all except these
    python run_batch_ollama.py --yes              # skip cost prompt
    python run_batch_ollama.py --skip-preflight   # not recommended

Output layout:
    ollama_flow/outputs/<ts>_batch[_pilot]/
      ├── overview.html
      ├── batch_manifest.json
      └── <scenario_id>/
            ├── 01_scenario.yaml
            ├── ...
            └── chain.html
"""

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Path setup — same as run_ollama.py
OLLAMA_FLOW_ROOT = Path(__file__).resolve().parent
PARENT_REPO_ROOT = OLLAMA_FLOW_ROOT.parent
sys.path.insert(0, str(OLLAMA_FLOW_ROOT))
sys.path.insert(0, str(PARENT_REPO_ROOT))

# Import run_ollama first — it monkey-patches db.DB_PATH at module load.
# All subsequent db calls (here and from process_scenario) use the
# alluvi_ollama.db path.
from run_ollama import process_scenario, _load_config, PLAN_LABEL, OUTPUT_ROOT
from src import db as _db
from src import scenario_loader, trace_html

# overview_html generator (parent's). May not match exact signature — handled
# below with multi-attempt fallback.
try:
    from src import overview_html
except ImportError:
    overview_html = None


# Estimated wall time per scenario on Blackwell (load + infer + unload, 3 stages)
DEFAULT_WALL_TIME_PER_SCENARIO_S = 300  # 5 min average

PILOT_COUNT = 5
POD_HOURLY_USD = 1.89


# ──────────────────────────────────────────────────────────────────────────
# Filtering
# ──────────────────────────────────────────────────────────────────────────

def _filter_scenarios(all_scenarios, only=None, exclude=None, pilot=False):
    """Apply --only / --exclude / --pilot in that order."""
    if only:
        only_set = {x.strip() for x in only.split(",") if x.strip()}
        filtered = [s for s in all_scenarios if s.get("id") in only_set]
        missing = only_set - {s.get("id") for s in filtered}
        if missing:
            print(f"[batch_ollama] WARNING: --only IDs not found: {sorted(missing)}")
    else:
        filtered = list(all_scenarios)

    if exclude:
        exclude_set = {x.strip() for x in exclude.split(",") if x.strip()}
        filtered = [s for s in filtered if s.get("id") not in exclude_set]

    if pilot:
        filtered = filtered[:PILOT_COUNT]

    return filtered


# ──────────────────────────────────────────────────────────────────────────
# Cost confirmation
# ──────────────────────────────────────────────────────────────────────────

def _confirm_cost(n_scenarios, skip=False) -> bool:
    """
    Ollama is free, so no LLM cost. Only GPU time is real.
    """
    wall_seconds = n_scenarios * DEFAULT_WALL_TIME_PER_SCENARIO_S
    wall_hours = wall_seconds / 3600
    gpu_cost_total = wall_hours * POD_HOURLY_USD

    print("")
    print("=" * 72)
    print(" BATCH PLAN (Ollama mode — LLM is free)")
    print("=" * 72)
    print(f"  scenarios:           {n_scenarios}")
    print(f"  LLM cost:            $0.00  (Ollama runs locally)")
    print(f"  Wall time (est):     ~{wall_seconds/60:.0f} min  (~{DEFAULT_WALL_TIME_PER_SCENARIO_S}s/scenario)")
    print(f"  GPU cost (est):      ~${gpu_cost_total:.2f}  ({wall_hours:.2f}h × ${POD_HOURLY_USD}/hr)")
    print("=" * 72)
    print("")

    if skip:
        print("[batch_ollama] --yes flag: skipping confirmation")
        return True

    try:
        response = input("Proceed? [y/N]: ").strip().lower()
    except EOFError:
        response = ""
    return response in ("y", "yes")


# ──────────────────────────────────────────────────────────────────────────
# Summary + manifest
# ──────────────────────────────────────────────────────────────────────────

def _build_summary(records, elapsed_s) -> dict:
    n_total = len(records)
    n_success = sum(1 for r in records if r.get("final_status") == "success")
    n_failed = sum(1 for r in records if r.get("final_status") == "failed")

    # GPU cost only (Ollama is free)
    total_cost = 0.0
    for r in records:
        for stage in ("step_1_meta", "step_2_meta", "step_3_meta"):
            meta = r.get(stage)
            if isinstance(meta, dict) and not meta.get("error"):
                total_cost += float(meta.get("cost_usd") or 0.0)

    # Stage 3 tally
    n_step3_ran = 0
    n_step3_failed = 0
    for r in records:
        s3 = r.get("step_3_meta")
        if isinstance(s3, dict):
            if s3.get("error"):
                n_step3_failed += 1
            elif s3:
                n_step3_ran += 1
    n_step3_skipped = n_total - n_step3_ran - n_step3_failed

    return {
        "total": n_total,
        "successful": n_success,
        "qc_failed": 0,  # no QC in Ollama flow
        "failed": n_failed,
        "total_cost_usd": round(total_cost, 3),
        "duration_seconds": int(elapsed_s),
        "step_3_ran": n_step3_ran,
        "step_3_failed": n_step3_failed,
        "step_3_skipped": n_step3_skipped,
    }


def _write_manifest(output_dir, run_id, records, summary, started_at, finished_at) -> None:
    manifest = {
        "run_id": run_id,
        "plan": PLAN_LABEL,
        "started_at": started_at,
        "finished_at": finished_at,
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


def _write_overview_html(output_dir, records, summary) -> None:
    if overview_html is None:
        print("[batch_ollama] overview_html module not importable — skipping overview.html")
        return

    attempts = [
        lambda: overview_html.write_overview_html(output_dir, records, summary),
        lambda: overview_html.write_overview(output_dir, records, summary),
        lambda: overview_html.write_overview_html(output_dir, records),
        lambda: overview_html.write_overview(output_dir, records),
        lambda: overview_html.generate(output_dir, records, summary),
        lambda: overview_html.generate(output_dir, records),
    ]
    last_error = None
    for fn in attempts:
        try:
            fn()
            return
        except (AttributeError, TypeError) as e:
            last_error = e
            continue
        except Exception as e:
            print(f"[batch_ollama] overview.html generator raised: {type(e).__name__}: {e}")
            return
    print(
        f"[batch_ollama] WARNING: no matching overview_html signature. "
        f"Last error: {last_error}. overview.html not generated."
    )


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Alluvi LOCAL image pipeline in Ollama mode (no Anthropic) "
            "for a batch of scenarios. Per-scenario flow: each scenario goes "
            "through all 3 stages before moving to the next."
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
        preflight_path = OLLAMA_FLOW_ROOT / "preflight_ollama.py"
        if preflight_path.exists():
            try:
                # Import dynamically — preflight_ollama lives at sibling
                import importlib.util
                spec = importlib.util.spec_from_file_location("preflight_ollama", preflight_path)
                pf = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(pf)
                if hasattr(pf, "run_preflight"):
                    errors, _warnings = pf.run_preflight(verbose=True)
                    if errors:
                        print("[batch_ollama] preflight failed — aborting batch")
                        return 1
                else:
                    print("[batch_ollama] preflight_ollama.py has no run_preflight() — skipping")
            except Exception as e:
                print(f"[batch_ollama] preflight crashed: {e} — continuing (use --skip-preflight to silence)")
        else:
            # Fall back to parent preflight if Ollama-specific one doesn't exist
            try:
                from preflight import run_preflight
                errors, _warnings = run_preflight(verbose=True)
                if errors:
                    print("[batch_ollama] parent preflight failed — aborting batch")
                    return 1
            except Exception as e:
                print(f"[batch_ollama] no preflight available ({e}) — continuing")

    # 2. Load + filter scenarios
    try:
        all_scenarios = scenario_loader.load_scenarios()
    except Exception as e:
        print(f"[batch_ollama] failed to load scenarios.yaml: {type(e).__name__}: {e}")
        return 1

    scenarios = _filter_scenarios(
        all_scenarios, only=args.only, exclude=args.exclude, pilot=args.pilot
    )
    if not scenarios:
        print("[batch_ollama] no scenarios after filters — nothing to do")
        return 1

    # 3. Config + cost confirmation
    config = _load_config()

    if not _confirm_cost(len(scenarios), skip=args.yes):
        print("[batch_ollama] aborted by user")
        return 1

    # 4. Output dir + DB run row
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pilot_suffix = "_pilot" if args.pilot else ""
    run_id = f"{timestamp}_batch{pilot_suffix}_ollama"
    output_dir = OUTPUT_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    notes_parts = [f"{len(scenarios)} scenarios"]
    if args.pilot:
        notes_parts.append("pilot mode")
    if args.only:
        notes_parts.append(f"only={args.only}")
    if args.exclude:
        notes_parts.append(f"exclude={args.exclude}")
    notes = "batch (ollama, local): " + ", ".join(notes_parts)

    try:
        _db.create_run(
            run_id=run_id,
            plan=PLAN_LABEL,
            pilot_mode=args.pilot,
            notes=notes,
        )
    except Exception as e:
        print(f"[batch_ollama] DB create_run failed: {e}")
        return 1

    # 5. Run the batch
    print("")
    print("=" * 72)
    print(f" ALLUVI — BATCH RUN (Ollama mode, local pipeline)")
    print(f" Run id:    {run_id}")
    print(f" Scenarios: {len(scenarios)}")
    print(f" Output:    {output_dir}")
    print(f" DB:        {_db.DB_PATH}")
    print("=" * 72)

    started_at = datetime.utcnow().isoformat()
    started = time.time()
    records: list[dict] = []
    interrupted = False

    try:
        for i, scenario in enumerate(scenarios, start=1):
            scenario_id = scenario.get("id", "?")
            print("")
            print(f"[batch_ollama] [{i}/{len(scenarios)}] === {scenario_id} ===")
            scenario_output_dir = output_dir / scenario_id

            try:
                record = process_scenario(
                    scenario, scenario_output_dir, config, run_id
                )
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(
                    f"[batch_ollama] {scenario_id}: UNEXPECTED uncaught exception: "
                    f"{type(e).__name__}: {e}"
                )
                traceback.print_exc()
                record = {
                    "scenario": scenario,
                    "output_dir": str(scenario_output_dir),
                    "final_status": "failed",
                    "error_stage": "uncaught",
                    "error_message": f"{type(e).__name__}: {e}",
                }

            # process_scenario writes chain.html with persona path
            # "../../../assets/persona.jpg" — correct for single-scenario
            # layout (ollama_flow/outputs/<ts>_<sid>/chain.html). For batch
            # layout (ollama_flow/outputs/<ts>_batch/<sid>/chain.html) the
            # chain.html lives one level deeper, so we re-write with
            # "../../../../assets/persona.jpg".
            try:
                trace_html.write_chain_html(
                    scenario_output_dir, record,
                    persona_rel_path="../../../../assets/persona.jpg",
                )
            except Exception as e:
                print(f"[batch_ollama] {scenario_id}: chain.html re-write failed (non-fatal): {e}")

            records.append(record)

            status = record.get("final_status", "?")
            s1 = (record.get("step_1_meta") or {}).get("elapsed_seconds", 0) or 0
            s2 = (record.get("step_2_meta") or {}).get("elapsed_seconds", 0) or 0
            s3_meta = record.get("step_3_meta")
            s3 = (s3_meta or {}).get("elapsed_seconds", 0) if isinstance(s3_meta, dict) else 0
            print(
                f"[batch_ollama] [{i}/{len(scenarios)}] {scenario_id}: {status.upper()} "
                f"(s1={s1:.0f}s s2={s2:.0f}s s3={s3:.0f}s)"
            )

    except KeyboardInterrupt:
        interrupted = True
        print("\n[batch_ollama] interrupted by user — finalizing partial batch...")

    elapsed = time.time() - started
    finished_at = datetime.utcnow().isoformat()
    summary = _build_summary(records, elapsed)

    # 6. Finalize DB
    try:
        _db.finalize_run(
            run_id=run_id,
            total_scenarios=summary["total"],
            successful=summary["successful"],
            failed=summary["failed"],
            total_cost_usd=summary["total_cost_usd"],
            duration_seconds=summary["duration_seconds"],
        )
    except Exception as e:
        print(f"[batch_ollama] DB finalize_run failed (non-fatal): {e}")

    # 7. Write manifest + overview
    try:
        _write_manifest(output_dir, run_id, records, summary, started_at, finished_at)
    except Exception as e:
        print(f"[batch_ollama] batch_manifest.json write failed (non-fatal): {e}")

    _write_overview_html(output_dir, records, summary)

    # 8. Final summary
    print("")
    print("=" * 72)
    print(" BATCH COMPLETE" + (" (interrupted)" if interrupted else ""))
    print("=" * 72)
    print(f"  total:        {summary['total']}")
    print(f"  ✓ success:    {summary['successful']}")
    print(f"  ✗ failed:     {summary['failed']}")
    print(f"  cost:         $0.00  (LLM free via Ollama)")
    print(f"  wall time:    {summary['duration_seconds']/60:.1f} min")
    gpu_cost = (summary['duration_seconds'] / 3600) * POD_HOURLY_USD
    print(f"  gpu cost:     ~${gpu_cost:.2f}  ({summary['duration_seconds']/3600:.2f}h × ${POD_HOURLY_USD}/hr)")
    print("")
    print(f"  Stage 3 (Kontext):")
    print(f"    ran:        {summary['step_3_ran']}")
    print(f"    failed:     {summary['step_3_failed']}")
    print(f"    skipped:    {summary['step_3_skipped']}")
    print("")
    print(f"  overview.html:    {output_dir / 'overview.html'}")
    print(f"  manifest:         {output_dir / 'batch_manifest.json'}")
    print(f"  DB run_id:        {run_id}")
    print("")

    if interrupted:
        return 130
    if summary["failed"] > 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())