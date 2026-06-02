#!/usr/bin/env python
"""
master.py — ALLUVI end-to-end runner: accounts -> images -> videos.

Runs the two pipelines as separate sequential subprocesses (so the image model
stack fully frees the GPU before the video stack loads):

  PHASE 1  supabase_pipeline/run_supabase_resident.py   (persona + N Qwen images/account)
  PHASE 2  video_pipeline/run_video.py                  (script->TTS->Wan->LatentSync)

Account scope (resolved once, passed to BOTH phases):
  --accounts a,b   -> exactly those tiktok_ids        (overrides everything)
  --all-accounts   -> every account                   (old reuse persona, new get one)
  (neither)        -> only accounts with NO persona    (new sign-ups; image pipeline default)

  --num N          -> N images per account, then N videos per account

Examples:
  python master.py --all-accounts --num 10 --yes      # everyone, 10 each
  python master.py --num 5                             # only new accounts, 5 each
  python master.py --accounts emma.callahan --num 2    # just Emma, 2
  python master.py --all-accounts --num 1 --dry-run    # preview, run nothing
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from supabase_pipeline import supabase_db

PY = sys.executable
IMAGE_SCRIPT = "supabase_pipeline/run_supabase_resident.py"
VIDEO_SCRIPT = "video_pipeline/run_video.py"
MULTISHOT_SCRIPT = "video_pipeline/multishot/run_multishot.py"
SILENTFIRST_SCRIPT = "video_pipeline/silentfirst/build_silentfirst.py"


def _all_id_map() -> dict:
    """Map every stored tiktok_id (and @/no-@ variants) -> the stored form."""
    m = {}
    for a in (supabase_db.get_all_accounts() or []):
        tid = a.get("tiktok_id")
        if not tid:
            continue
        bare = tid.lstrip("@")
        m[tid] = tid
        m[bare] = tid
        m["@" + bare] = tid
    return m


def resolve_accounts(args) -> tuple[list[str], str, list[str]]:
    """Return (tiktok_ids, mode_label, missing)."""
    if args.accounts:
        typed = [x.strip() for x in args.accounts.split(",") if x.strip()]
        idmap = _all_id_map()
        resolved, missing = [], []
        for t in typed:
            hit = idmap.get(t) or idmap.get(t.lstrip("@")) or idmap.get("@" + t.lstrip("@"))
            if hit:
                resolved.append(hit)
            else:
                missing.append(t)
        return resolved, "explicit accounts", missing
    if args.all_accounts:
        accts = supabase_db.get_all_accounts() or []
        return [a["tiktok_id"] for a in accts], "ALL accounts", []
    accts = supabase_db.get_accounts_without_persona() or []
    return [a["tiktok_id"] for a in accts], "NEW accounts only (no persona yet)", []


def run_phase(title: str, cmd: list[str]) -> int:
    print(f"\n{'='*64}\n{title}\n  $ {PY} {' '.join(cmd)}\n{'='*64}", flush=True)
    t0 = time.time()
    rc = subprocess.run([PY] + cmd, cwd=str(REPO_ROOT)).returncode
    print(f"\n[{title}] finished rc={rc} in {time.time()-t0:.0f}s", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser(description="ALLUVI master runner (images -> videos)")
    ap.add_argument("--num", type=int, default=1,
                    help="images per account AND videos per account (default 1)")
    ap.add_argument("--all-accounts", action="store_true",
                    help="process every account (default: only accounts without a persona)")
    ap.add_argument("--accounts", default=None,
                    help="comma-separated tiktok_ids; overrides --all-accounts and the default")
    ap.add_argument("--yes", action="store_true",
                    help="skip the confirmation prompt (also auto-confirms both phases)")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="pass --skip-preflight to the image phase")
    ap.add_argument("--skip-images", action="store_true",
                    help="skip Phase 1 (use existing images only)")
    ap.add_argument("--skip-videos", action="store_true",
                    help="skip Phase 2 (only make images)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the exact phase commands, run nothing")
    ap.add_argument("--video-mode", choices=["single", "multishot", "silentfirst"],
                    default="single", help="phase-2 video flow (default single 5s)")
    ap.add_argument("--num-shots", type=int, default=8,
                    help="shots/beats per video for multishot & silentfirst (default 8)")
    ap.add_argument("--shot-seconds", type=int, default=5,
                    help="seconds per shot/beat for multishot & silentfirst (default 5)")
    ap.add_argument("--tail-seconds", type=float, default=2.0,
                    help="held-frame + silence tail after audio for multishot & silentfirst (default 2.0)")
    args = ap.parse_args()

    ids, mode, missing = resolve_accounts(args)
    if missing:
        print(f"WARNING — these typed accounts were not found and will be ignored: {missing}")
    if not ids:
        print(f"Nothing to do — no accounts match ({mode}).")
        return

    n = args.num
    # Build the two phase commands. The resolved id list is passed explicitly to
    # BOTH, so the scope is identical in each phase regardless of mode.
    acct_csv = ",".join(ids)
    image_cmd = [IMAGE_SCRIPT, "--accounts", acct_csv, "--num-scenarios", str(n), "--yes"]
    if args.skip_preflight:
        image_cmd.append("--skip-preflight")
    if args.video_mode == "multishot":
        video_cmd = [MULTISHOT_SCRIPT, "--accounts", acct_csv, "--num-shots", str(args.num_shots), "--shot-seconds", str(args.shot_seconds), "--tail-seconds", str(args.tail_seconds), "--yes"]
    elif args.video_mode == "silentfirst":
        video_cmd = [SILENTFIRST_SCRIPT, "--accounts", acct_csv, "--num-shots", str(args.num_shots), "--shot-seconds", str(args.shot_seconds), "--tail-seconds", str(args.tail_seconds), "--yes"]
    else:
        video_cmd = [VIDEO_SCRIPT, "--accounts", acct_csv, "--num-videos", str(n), "--yes"]

    print(f"\nMASTER PLAN")
    print(f"  scope      : {mode}")
    print(f"  accounts   : {len(ids)}  -> {acct_csv}")
    print(f"  per account: {n} image(s) then {n} video(s)")
    print(f"  video mode : {args.video_mode}" + ("" if args.video_mode == "single" else f"  ({args.num_shots} x {args.shot_seconds}s)"))
    print(f"  ceiling     : up to {len(ids)*n} images then up to {len(ids)*n} videos")
    print(f"  phase 1    : {'SKIPPED' if args.skip_images else 'run image pipeline'}")
    print(f"  phase 2    : {'SKIPPED' if args.skip_videos else 'run video pipeline'}")
    print(f"\n  PHASE 1 CMD: {PY} {' '.join(image_cmd)}")
    print(f"  PHASE 2 CMD: {PY} {' '.join(video_cmd)}")

    if args.dry_run:
        print("\n(dry-run) nothing executed.")
        return
    if not args.yes:
        if input("\nProceed with the full run? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return

    t0 = time.time()
    if not args.skip_images:
        rc1 = run_phase("PHASE 1 — IMAGES", image_cmd)
        if rc1 != 0:
            print("WARNING — image phase returned an error; continuing to video phase "
                  "(any images that did finish can still be turned into videos).")
    else:
        print("\n[PHASE 1] skipped (--skip-images)")

    if not args.skip_videos:
        run_phase("PHASE 2 — VIDEOS", video_cmd)
    else:
        print("\n[PHASE 2] skipped (--skip-videos)")

    print(f"\n{'#'*64}\nMASTER DONE in {time.time()-t0:.0f}s "
          f"({mode}, {len(ids)} account(s), n={n}).\n{'#'*64}")


if __name__ == "__main__":
    main()
