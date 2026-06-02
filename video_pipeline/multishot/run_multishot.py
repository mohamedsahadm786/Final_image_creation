#!/usr/bin/env python
"""
video_pipeline/multishot/run_multishot.py — build one multi-shot video.

Method (research-backed): one finished realism photo is the FIRST FRAME of every
shot (identity/wardrobe/background pixel-identical) -> per shot: F5 voice (fixed
seed+voice for consistency) -> Wan motion (frames from that shot's audio) ->
LatentSync -> then ffmpeg stitch into one clip. Reuses step_4/5/6 unchanged.

v1 writes a local manifest.json per video; Supabase audit tables come next.

  python video_pipeline/multishot/run_multishot.py --accounts emma.callahan --num-shots 2 --yes
"""
from __future__ import annotations
import argparse
import datetime
import json
import random
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from supabase_pipeline import supabase_db
from video_pipeline import step_4_tts_f5 as tts
from video_pipeline import step_5_video_wan as wan
from video_pipeline import step_6_lipsync as lip
from video_pipeline.multishot import script_builder_multi
from video_pipeline.multishot import stitch as stitcher
from video_pipeline import extend_tail

VOICE_BY_GENDER = {"female": "voices_examples/female/female_02.wav",
                   "male": "voices_examples/male/male_01.wav"}


def _voice_for(account: dict) -> str:
    g = (account.get("gender") or "").strip().lower()
    return VOICE_BY_GENDER["male"] if g.startswith("m") else VOICE_BY_GENDER["female"]


def _scenario_for(scenario_id: str) -> dict:
    try:
        from src import scenario_loader
        for s in (scenario_loader.load_scenarios() or []):
            if s.get("id") == scenario_id:
                return s
    except Exception:
        pass
    return {"id": scenario_id}


def _finished_output(persona_id: int, output_id: int | None):
    rows = (supabase_db.client().table("outputs").select("*")
            .eq("persona_id", persona_id).eq("status", "done").execute().data or [])
    rows = [r for r in rows if r.get("final_image_local_path")]
    if output_id:
        rows = [r for r in rows if r["id"] == output_id]
    rows.sort(key=lambda r: r["id"])
    return rows[0] if rows else None


def _resolve_accounts(accounts_arg):
    if accounts_arg:
        variants = []
        for w in [x.strip() for x in accounts_arg.split(",") if x.strip()]:
            bare = w.lstrip("@")
            variants += [bare, "@" + bare]
        return supabase_db.get_accounts_by_tiktok_ids(list(dict.fromkeys(variants))) or []
    return supabase_db.get_all_accounts() or []


def build_one(account, output, num_shots, base_seed, run_dir, handles, target_seconds=10, tail_seconds=2.0):
    tts_h, wan_h, lip_h = handles
    sid = output["scenario_id"]
    image_path = output["final_image_local_path"]
    voice = _voice_for(account)
    scenario = _scenario_for(sid)

    out_dir = run_dir / (account.get("tiktok_id") or "acct").lstrip("@") / sid
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {account.get('tiktok_id')} / {sid} : {num_shots} shots ===")
    print(f"    anchor image: {image_path}")
    script = script_builder_multi.build_multishot_script(account, scenario, num_shots=num_shots, target_seconds=target_seconds)
    shots = script["shots"]

    final_clips = []
    shot_records = []
    for i, sh in enumerate(shots):
        tag = f"{sid}#shot{i+1}"
        shot_dir = out_dir / f"shot_{i+1}"
        shot_dir.mkdir(parents=True, exist_ok=True)

        a = tts.generate(tts_h, sh["dialogue"], shot_dir / "audio",
                         narrator_voice=voice, seed=base_seed, scene_id=tag)
        frame_cap = int((target_seconds + 3) * 16)
        n_frames = wan.frames_for_duration(a["audio_duration_seconds"], cap=frame_cap)
        v = wan.generate(wan_h, image_path, shot_dir / "silent",
                         motion_prompt=sh["wan_motion_prompt"],
                         negative_prompt=sh.get("wan_negative_prompt"),
                         num_frames=n_frames,
                         seed=base_seed + 1 + i, scene_id=tag)
        f = lip.generate(lip_h, v["local_path"], a["local_path"], shot_dir / "final",
                         scene_id=tag)
        final_clips.append(f["local_path"])
        shot_records.append({
            "index": i + 1, "dialogue": sh["dialogue"],
            "audio": a["local_path"], "audio_seconds": a["audio_duration_seconds"],
            "silent": v["local_path"], "num_frames": v["num_frames"], "fps": v["fps"],
            "final": f["local_path"], "audio_seed": base_seed, "video_seed": base_seed + 1 + i,
        })

    stitched = out_dir / "stitched.mp4"
    stitcher.stitch(final_clips, stitched, punch_in=1.2)
    if tail_seconds and tail_seconds > 0:
        extend_tail.extend_tail(stitched, seconds=tail_seconds)
        print(f"    +{tail_seconds:.1f}s natural tail (held frame + silence)")

    manifest = {
        "account": account.get("tiktok_id"), "scenario_id": sid,
        "source_output_id": output["id"], "anchor_image": image_path,
        "num_shots": len(shots), "narrative_theme": script.get("narrative_theme"),
        "continuity_block": script.get("continuity_block"),
        "language": script.get("language"), "voice": voice, "base_seed": base_seed,
        "shots": shot_records, "stitched_video": str(stitched),
        "created_at": datetime.datetime.now().isoformat(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return str(stitched)


def main():
    ap = argparse.ArgumentParser(description="ALLUVI multi-shot video builder")
    ap.add_argument("--accounts", default=None, help="comma-separated tiktok_ids (omit = all)")
    ap.add_argument("--num-shots", type=int, default=2, help="shots to stitch (default 2 ~= 10s)")
    ap.add_argument("--output-id", type=int, default=None, help="pin a specific finished output")
    ap.add_argument("--seed", type=int, default=None, help="base seed (default random)")
    ap.add_argument("--shot-seconds", type=int, default=10, help="target seconds per shot (default 10)")
    ap.add_argument("--tail-seconds", type=float, default=2.0, help="held-frame + silence tail after audio (default 2.0)")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    accounts = _resolve_accounts(args.accounts)
    if not accounts:
        print("No matching accounts."); return

    work = []
    for acct in accounts:
        persona = supabase_db.get_persona_for_account(acct["id"])
        if not persona or not persona.get("id"):
            continue
        out = _finished_output(persona["id"], args.output_id)
        if out:
            work.append((acct, out))
    if not work:
        print("No finished realism images found for the selected account(s). "
              "Run the image pipeline first."); return

    base_seed = args.seed if args.seed is not None else random.randint(1, 2_000_000_000)
    print(f"\nMulti-shot plan ({args.num_shots} shots each, base_seed={base_seed}):")
    for acct, out in work:
        print(f"  - {acct.get('tiktok_id')}  scene={out['scenario_id']}  output_id={out['id']}")
    print(f"  ~{args.shot_seconds}s/shot, est. ~{args.num_shots * (13 if args.shot_seconds <= 5 else 25)} min at full quality")
    if not args.yes and input("\nProceed? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Aborted."); return

    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_multishot")
    run_dir = REPO_ROOT / "outputs" / run_id
    print("\n[run_multishot] warming up ComfyUI servers...")
    handles = (tts.load_pipeline(), wan.load_pipeline(), lip.load_pipeline())

    ok = 0
    for acct, out in work:
        try:
            path = build_one(acct, out, args.num_shots, base_seed, run_dir, handles, target_seconds=args.shot_seconds, tail_seconds=args.tail_seconds)
            ok += 1
            print(f"--- DONE {acct.get('tiktok_id')} -> {path}")
        except Exception as e:
            print(f"!!! FAILED {acct.get('tiktok_id')}: {e}")
            traceback.print_exc()

    tts.unload_pipeline(handles[0]); wan.unload_pipeline(handles[1]); lip.unload_pipeline(handles[2])
    print(f"\n{'='*60}\nFinished: {ok}/{len(work)} multi-shot video(s).")
    print(f"Run folder: outputs/{run_id}/\n{'='*60}")


if __name__ == "__main__":
    main()
