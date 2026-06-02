#!/usr/bin/env python
"""
video_pipeline/silentfirst/build_silentfirst.py — build one SILENT-FIRST video.
Sibling of run_multishot. Method: full audio FIRST (intro silence + speech + outro
silence) -> render silent Wan clips -> dHash frame-join (punch-in fallback) -> top-up
loop until silent >= audio -> ONE LatentSync pass. The opening/closing silence gives a
natural non-speaking start/end with REAL motion (mouth closed), so no frozen tail is needed.
  python video_pipeline/silentfirst/build_silentfirst.py --accounts emma.callahan --num-shots 8 --shot-seconds 5 --yes
"""
from __future__ import annotations
import argparse, datetime, json, random, subprocess, sys, traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from supabase_pipeline import supabase_db
from video_pipeline import step_4_tts_f5 as tts
from video_pipeline import step_5_video_wan as wan
from video_pipeline import step_6_lipsync as lip
from video_pipeline import extend_tail
from video_pipeline.multishot import script_builder_multi
from video_pipeline.silentfirst import frame_join

VOICE_BY_GENDER = {"female": "voices_examples/female/female_02.wav",
                   "male": "voices_examples/male/male_01.wav"}
_NULL = subprocess.DEVNULL


def _voice_for(account):
    g = (account.get("gender") or "").strip().lower()
    return VOICE_BY_GENDER["male"] if g.startswith("m") else VOICE_BY_GENDER["female"]


def _scenario_for(scenario_id):
    try:
        from src import scenario_loader
        for s in (scenario_loader.load_scenarios() or []):
            if s.get("id") == scenario_id:
                return s
    except Exception:
        pass
    return {"id": scenario_id}


def _finished_output(persona_id, output_id):
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
            bare = w.lstrip("@"); variants += [bare, "@" + bare]
        return supabase_db.get_accounts_by_tiktok_ids(list(dict.fromkeys(variants))) or []
    return supabase_db.get_all_accounts() or []


def _dur(p):
    return float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(p)]).decode())


def _concat_audio(paths, out_wav, intro=0.0, outro=0.0):
    """Concatenate the beat audios, then pad leading/trailing SILENCE (no music)."""
    out_wav = Path(out_wav)
    tmp = out_wav.with_suffix(".speech.wav")
    inp = []
    for p in paths:
        inp += ["-i", str(p)]
    filt = "".join(f"[{k}:a]" for k in range(len(paths))) + f"concat=n={len(paths)}:v=0:a=1[a]"
    subprocess.run(["ffmpeg", "-y"] + inp + ["-filter_complex", filt, "-map", "[a]",
                    "-ar", "16000", "-ac", "1", str(tmp)], check=True, stdout=_NULL, stderr=_NULL)
    af = []
    if intro and intro > 0:
        af.append(f"adelay={int(intro * 1000)}")
    if outro and outro > 0:
        af.append(f"apad=pad_dur={outro}")
    if af:
        subprocess.run(["ffmpeg", "-y", "-i", str(tmp), "-af", ",".join(af),
                        "-ar", "16000", "-ac", "1", str(out_wav)], check=True, stdout=_NULL, stderr=_NULL)
        tmp.unlink()
    else:
        tmp.replace(out_wav)


def build_one(account, output, num_beats, base_seed, run_dir, handles,
              target_seconds=5, threshold=70, max_iters=8,
              intro_seconds=2.0, outro_seconds=2.0, tail_seconds=0.0,
              lips_expression=2.0, inference_steps=40, punch_in=1.2):
    tts_h, wan_h, lip_h = handles
    sid = output["scenario_id"]; image_path = output["final_image_local_path"]
    voice = _voice_for(account); scenario = _scenario_for(sid)
    out_dir = run_dir / (account.get("tiktok_id") or "acct").lstrip("@") / sid
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {account.get('tiktok_id')} / {sid} : silent-first, {num_beats} beats ===")
    print(f"    anchor image: {image_path}")
    script = script_builder_multi.build_multishot_script(
        account, scenario, num_shots=num_beats, target_seconds=target_seconds)
    beats = script["shots"]
    motions = [b["wan_motion_prompt"] for b in beats]
    negs = [b.get("wan_negative_prompt") for b in beats]

    # 1) FULL audio = intro silence + concatenated beats + outro silence
    adir = out_dir / "audio"; adir.mkdir(parents=True, exist_ok=True)
    apaths = []
    for i, b in enumerate(beats):
        a = tts.generate(tts_h, b["dialogue"], adir / f"a{i+1}",
                         narrator_voice=voice, seed=base_seed, scene_id=f"{sid}#a{i+1}")
        apaths.append(a["local_path"])
    full_audio = out_dir / "full_audio.wav"
    _concat_audio(apaths, full_audio, intro=intro_seconds, outro=outro_seconds)
    A = _dur(full_audio)
    print(f"    full audio = {A:.2f}s  (incl {intro_seconds:.0f}s intro + {outro_seconds:.0f}s outro silence)")

    # 2) silent footage + top-up loop until silent >= audio
    sdir = out_dir / "silent"; sdir.mkdir(parents=True, exist_ok=True); silent = []
    def render(idx, motion, neg):
        nf = wan.frames_for_duration(target_seconds + 0.5, cap=int((target_seconds + 3) * 16))
        v = wan.generate(wan_h, image_path, sdir / f"s{idx}", motion_prompt=motion,
                         negative_prompt=neg, num_frames=nf,
                         seed=base_seed + 100 + idx, scene_id=f"{sid}#s{idx}")
        return v["local_path"]
    for i in range(num_beats):
        silent.append(render(i + 1, motions[i], negs[i]))
    joined = out_dir / "silent_joined.mp4"; report = []; it = 0
    while True:
        it += 1
        _, report = frame_join.join(silent, joined, threshold=threshold, punch_in=punch_in)
        V = _dur(joined); print(f"    join iter {it}: {len(silent)} clips -> {V:.2f}s (need {A:.2f}s)")
        if V >= A or it >= max_iters:
            break
        yp = max(0.5, V / len(silent)); more = max(1, int((A - V) / yp) + 1)
        print(f"    short {A - V:.2f}s -> rendering {more} more clip(s)")
        for _ in range(more):
            idx = len(silent) + 1
            silent.append(render(idx, motions[idx % len(motions)], negs[idx % len(negs)]))

    # 3) ONE lip-sync over the whole video (auto-trims to audio length)
    f = lip.generate(lip_h, str(joined), str(full_audio), out_dir / "final",
                     lips_expression=lips_expression, inference_steps=inference_steps,
                     scene_id=f"{sid}#final")
    final_path = f["local_path"]

    # frozen-frame tail ONLY if there is no outro silence (the silence gives a real-motion ending)
    if tail_seconds and tail_seconds > 0 and not (outro_seconds and outro_seconds > 0):
        final_path = extend_tail.extend_tail(final_path, seconds=tail_seconds)
        print(f"    +{tail_seconds:.1f}s held-frame tail")

    (out_dir / "manifest.json").write_text(json.dumps({
        "method": "silentfirst", "account": account.get("tiktok_id"), "scenario_id": sid,
        "source_output_id": output["id"], "anchor_image": image_path, "num_beats": num_beats,
        "num_silent_clips": len(silent), "full_audio_seconds": A,
        "intro_seconds": intro_seconds, "outro_seconds": outro_seconds,
        "silent_joined_seconds": _dur(joined), "narrative_theme": script.get("narrative_theme"),
        "language": script.get("language"), "voice": voice, "base_seed": base_seed,
        "threshold": threshold, "seam_report": report, "final_video": final_path,
        "created_at": datetime.datetime.now().isoformat(),
    }, indent=2, ensure_ascii=False))
    return final_path


def main():
    ap = argparse.ArgumentParser(description="ALLUVI silent-first video builder")
    ap.add_argument("--accounts", default=None, help="comma-separated tiktok_ids (omit = all)")
    ap.add_argument("--num-shots", type=int, default=8, help="dialogue beats (default 8)")
    ap.add_argument("--shot-seconds", type=int, default=5, help="seconds per beat/clip (default 5)")
    ap.add_argument("--output-id", type=int, default=None, help="pin a specific finished output")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--threshold", type=int, default=70, help="dHash frame-match threshold (default 70)")
    ap.add_argument("--intro-seconds", type=float, default=2.0, help="opening silence pause (default 2.0)")
    ap.add_argument("--outro-seconds", type=float, default=2.0, help="closing silence pause (default 2.0)")
    ap.add_argument("--tail-seconds", type=float, default=0.0,
                    help="held-frame tail; ignored when outro-seconds > 0 (default 0.0)")
    ap.add_argument("--lips-expression", type=float, default=2.0,
                    help="LatentSync mouth-opening strength (default 2.0)")
    ap.add_argument("--inference-steps", type=int, default=40,
                    help="LatentSync steps; higher = cleaner mask seam (default 40)")
    ap.add_argument("--punch-in", type=float, default=1.2,
                    help="frame-join punch-in zoom on fallback seams (1.0=off; default 1.2)")
    ap.add_argument("--lips-expression", type=float, default=2.0,
                    help="LatentSync mouth-opening strength (default 2.0)")
    ap.add_argument("--inference-steps", type=int, default=40,
                    help="LatentSync steps; higher = cleaner mask seam (default 40)")
    ap.add_argument("--punch-in", type=float, default=1.2,
                    help="frame-join punch-in zoom on fallback seams (1.0=off; default 1.2)")
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
        print("No finished realism images found. Run the image pipeline first."); return

    base_seed = args.seed if args.seed is not None else random.randint(1, 2_000_000_000)
    print(f"\nSilent-first plan ({args.num_shots} beats each, base_seed={base_seed}):")
    for acct, out in work:
        print(f"  - {acct.get('tiktok_id')}  scene={out['scenario_id']}  output_id={out['id']}")
    print(f"  ~{args.shot_seconds}s/clip; {args.intro_seconds:.0f}s+{args.outro_seconds:.0f}s silence; "
          f"footage top-up until video >= audio")
    if not args.yes and input("\nProceed? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Aborted."); return

    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_silentfirst")
    run_dir = REPO_ROOT / "outputs" / run_id
    print("\n[build_silentfirst] warming up ComfyUI servers...")
    handles = (tts.load_pipeline(), wan.load_pipeline(), lip.load_pipeline())
    ok = 0
    for acct, out in work:
        try:
            path = build_one(acct, out, args.num_shots, base_seed, run_dir, handles,
                             target_seconds=args.shot_seconds, threshold=args.threshold,
                             intro_seconds=args.intro_seconds, outro_seconds=args.outro_seconds,
                             tail_seconds=args.tail_seconds, lips_expression=args.lips_expression,
                             inference_steps=args.inference_steps, punch_in=args.punch_in)
            ok += 1; print(f"--- DONE {acct.get('tiktok_id')} -> {path}")
        except Exception as e:
            print(f"!!! FAILED {acct.get('tiktok_id')}: {e}"); traceback.print_exc()
    tts.unload_pipeline(handles[0]); wan.unload_pipeline(handles[1]); lip.unload_pipeline(handles[2])
    print(f"\n{'='*60}\nFinished: {ok}/{len(work)} silent-first video(s).")
    print(f"Run folder: outputs/{run_id}/\n{'='*60}")


if __name__ == "__main__":
    main()
