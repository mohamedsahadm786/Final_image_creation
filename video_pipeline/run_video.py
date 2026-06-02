#!/usr/bin/env python
"""
video_pipeline/run_video.py — VIDEO orchestrator.

Consumes finished `outputs` rows (done + QC-acceptable, no video yet) and turns
each into a final lip-synced clip:

    script (Opus) -> Stage 4 voice (F5-TTS) -> Stage 5 video (Wan, frames from
    audio length) -> Stage 6 lip-sync (LatentSync) -> videos + audit rows.

Defaults:
  * no --accounts  -> ALL accounts (only those with finished images do work)
  * --num-videos N -> up to N videos PER account (default 1)
  * resume: any output that already has a video row is skipped

Examples:
  python video_pipeline/run_video.py                       # all accounts, 1 each
  python video_pipeline/run_video.py --accounts emma.callahan
  python video_pipeline/run_video.py --num-videos 2 --yes
  python video_pipeline/run_video.py --dry-run
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from supabase_pipeline import supabase_db
from video_pipeline import video_db, script_builder
from video_pipeline import step_4_tts_f5 as tts
from video_pipeline import step_5_video_wan as wan
from video_pipeline import step_6_lipsync as lip

SCRIPT_MODEL = "claude-opus-4-7"
# Reference voices by gender. female_02 is the proven default; the male path is a
# best-guess until verified (only female accounts have finished images for now).
VOICE_BY_GENDER = {
    "female": "voices_examples/female/female_02.wav",
    "male": "voices_examples/male/male_01.wav",
}
WELL_SUPPORTED_LANGS = {"english", "chinese", "mandarin"}


def _safe(tiktok_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", (tiktok_id or "acct").lstrip("@"))


def _db(fn, *args, **kwargs):
    """Run a DB write; never let an audit failure kill a render."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"   [db] {fn.__name__} failed (non-fatal): {e}")
        return None


def _load_scenarios_by_id() -> dict:
    try:
        from src import scenario_loader
        return {s.get("id"): s for s in (scenario_loader.load_scenarios() or [])}
    except Exception as e:
        print(f"[run_video] could not load scenarios.yaml ({e}); "
              f"using scenario_id only")
        return {}


def _select_accounts(accounts_arg: str | None) -> list[dict]:
    if accounts_arg:
        wanted = [x.strip() for x in accounts_arg.split(",") if x.strip()]
        variants = []
        for w in wanted:
            bare = w[1:] if w.startswith("@") else w
            variants += [bare, "@" + bare]
        variants = list(dict.fromkeys(variants))
        return supabase_db.get_accounts_by_tiktok_ids(variants) or []
    return supabase_db.get_all_accounts() or []


def _build_worklist(accounts: list[dict], num_videos: int,
                    include_qc_skipped: bool) -> list[tuple]:
    work = []
    for acct in accounts:
        persona = _db(supabase_db.get_persona_for_account, acct["id"])
        if not persona or not persona.get("id"):
            continue
        outs = video_db.get_outputs_needing_video(
            persona_ids=[persona["id"]],
            include_qc_skipped=include_qc_skipped,
            limit=num_videos,
        )
        for o in outs:
            work.append((acct, persona["id"], o))
    return work


def _process_one(acct, persona_id, output, scenarios_by_id, run_pk, handles) -> str:
    tts_h, wan_h, lip_h = handles
    sid = output["scenario_id"]
    oid = output["id"]
    video_id = None

    out_dir = REPO_ROOT / "outputs" / run_pk["run_id"] / _safe(acct.get("tiktok_id")) / sid
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id_pk = run_pk["pk"]

    # 1) SCRIPT (Opus) -------------------------------------------------------
    scenario = scenarios_by_id.get(sid) or {"id": sid}
    st = _db(supabase_db.start_stage, stage_name="video_script",
             run_pk=run_id_pk, persona_id=persona_id, output_id=oid)
    script = script_builder.build_script(acct, scenario)
    _db(supabase_db.insert_llm_call, purpose="video_script", model=SCRIPT_MODEL,
        stage_execution_id=st, system_prompt_name="master_prompt_script",
        system_prompt_version="v1", parsed_json=script)
    _db(supabase_db.finish_stage, st, status="done")

    dialogue = script["dialogue"]
    motion = script["wan_motion_prompt"]
    neg = script.get("wan_negative_prompt")
    language = (script.get("language") or acct.get("language") or "").strip()
    gender = (acct.get("gender") or "").strip().lower()
    if gender.startswith("m"):
        voice = VOICE_BY_GENDER["male"]
    else:
        voice = VOICE_BY_GENDER["female"]
    if language and language.lower() not in WELL_SUPPORTED_LANGS:
        print(f"   NOTE: language '{language}' is not native to F5TTS_v1_Base "
              f"(English/Chinese) — audio will be rough until the multilingual stage.")

    vrow = _db(video_db.upsert_video, output_id=oid, persona_id=persona_id, run_pk=run_id_pk,
               scenario_id=sid, dialogue=dialogue, language=language or None,
               hook_style=script.get("hook_style"), scene_mood=script.get("scene_mood"),
               wan_motion_prompt=motion, wan_negative_prompt=neg,
               ref_audio_path=voice, ref_text=None, status="running")
    if isinstance(vrow, dict):
        video_id = vrow.get("id")

    try:
        # 2) VOICE (F5-TTS) --------------------------------------------------
        st = _db(supabase_db.start_stage, stage_name="stage4_tts",
                 run_pk=run_id_pk, persona_id=persona_id, output_id=oid)
        a = tts.generate(tts_h, dialogue, out_dir / "audio",
                         narrator_voice=voice, scene_id=sid)
        _db(supabase_db.finish_stage, st, status="done", elapsed_seconds=a["elapsed_seconds"])
        _db(video_db.insert_media_generation, stage_name="stage4_tts", media_type="audio",
            video_id=video_id, stage_execution_id=st, model_name=a["model_name"],
            workflow_template=a["workflow_template"], comfyui_server=a["comfyui_server"],
            prompt=dialogue, params=a["params"], output_local_path=a["local_path"],
            duration_seconds=a["audio_duration_seconds"], elapsed_seconds=a["elapsed_seconds"])
        if video_id:
            _db(video_db.update_video, video_id, audio_local_path=a["local_path"],
                audio_duration_seconds=a["audio_duration_seconds"])

        # 3) VIDEO (Wan, frames from audio) ----------------------------------
        st = _db(supabase_db.start_stage, stage_name="stage5_video",
                 run_pk=run_id_pk, persona_id=persona_id, output_id=oid)
        v = wan.generate(wan_h, output["final_image_local_path"], out_dir / "silent",
                         motion_prompt=motion, negative_prompt=neg,
                         audio_duration_seconds=a["audio_duration_seconds"], scene_id=sid)
        _db(supabase_db.finish_stage, st, status="done", elapsed_seconds=v["elapsed_seconds"])
        _db(video_db.insert_media_generation, stage_name="stage5_video", media_type="video",
            video_id=video_id, stage_execution_id=st, model_name=v["model_name"],
            workflow_template=v["workflow_template"], comfyui_server=v["comfyui_server"],
            prompt=motion, negative_prompt=v["negative_prompt"], params=v["params"],
            input_paths=[output["final_image_local_path"]], output_local_path=v["local_path"],
            elapsed_seconds=v["elapsed_seconds"])
        if video_id:
            _db(video_db.update_video, video_id, silent_video_local_path=v["local_path"],
                num_frames=v["num_frames"], fps=v["fps"])

        # 4) LIP-SYNC (LatentSync) -------------------------------------------
        st = _db(supabase_db.start_stage, stage_name="stage6_lipsync",
                 run_pk=run_id_pk, persona_id=persona_id, output_id=oid)
        f = lip.generate(lip_h, v["local_path"], a["local_path"], out_dir / "final",
                         scene_id=sid)
        _db(supabase_db.finish_stage, st, status="done", elapsed_seconds=f["elapsed_seconds"])
        _db(video_db.insert_media_generation, stage_name="stage6_lipsync", media_type="video",
            video_id=video_id, stage_execution_id=st, model_name=f["model_name"],
            workflow_template=f["workflow_template"], comfyui_server=f["comfyui_server"],
            params=f["params"], input_paths=[v["local_path"], a["local_path"]],
            output_local_path=f["local_path"], elapsed_seconds=f["elapsed_seconds"])
        if video_id:
            _db(video_db.update_video, video_id,
                final_video_local_path=f["local_path"], status="done")

        return f["local_path"]
    except Exception:
        if video_id:
            _db(video_db.update_video, video_id, status="error")
        raise


def main():
    ap = argparse.ArgumentParser(description="ALLUVI video pipeline orchestrator")
    ap.add_argument("--accounts", default=None,
                    help="comma-separated tiktok ids (e.g. emma.callahan,liam.foster). "
                         "Omit = ALL accounts.")
    ap.add_argument("--all-accounts", action="store_true",
                    help="explicit 'all accounts' (same as omitting --accounts)")
    ap.add_argument("--num-videos", type=int, default=1,
                    help="max videos PER account this run (default 1)")
    ap.add_argument("--no-qc-skipped", action="store_true",
                    help="only use QC-passed outputs (default also allows QC-skipped)")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the work list and exit without rendering")
    args = ap.parse_args()

    accounts_arg = None if args.all_accounts else args.accounts
    accounts = _select_accounts(accounts_arg)
    if not accounts:
        print("No matching accounts found.")
        return

    work = _build_worklist(accounts, args.num_videos, not args.no_qc_skipped)
    if not work:
        print("Nothing to do — no finished scene images need a video "
              "(run the image pipeline first, or they may already have videos).")
        return

    print(f"\nVideo pipeline — {len(work)} clip(s) to create "
          f"(~9-10 min each at full quality):")
    for acct, _pid, o in work:
        print(f"  - {acct.get('tiktok_id'):<20} scene={o['scenario_id']:<32} output_id={o['id']}")

    if args.dry_run:
        print("\n(dry-run) exiting without rendering.")
        return
    if not args.yes:
        ans = input("\nProceed? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            return

    scenarios_by_id = _load_scenarios_by_id()
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_video")
    pk = _db(supabase_db.create_run, run_id=run_id, flow_name="video_pipeline",
             total_scenarios=len(work), notes="audio-first script->TTS->Wan->LatentSync")
    run_pk = {"run_id": run_id, "pk": pk if isinstance(pk, int) else None}

    print("\n[run_video] warming up ComfyUI servers...")
    handles = (tts.load_pipeline(), wan.load_pipeline(), lip.load_pipeline())

    ok = 0
    for acct, persona_id, output in work:
        tag = f"{acct.get('tiktok_id')}/{output['scenario_id']}"
        print(f"\n=== {tag}  (output_id={output['id']}) ===")
        try:
            final_path = _process_one(acct, persona_id, output, scenarios_by_id, run_pk, handles)
            ok += 1
            print(f"--- DONE {tag} -> {final_path}")
        except Exception as e:
            print(f"!!! FAILED {tag}: {e}")
            traceback.print_exc()

    tts.unload_pipeline(handles[0])
    wan.unload_pipeline(handles[1])
    lip.unload_pipeline(handles[2])
    if run_pk["pk"]:
        _db(supabase_db.finalize_run, run_pk["pk"],
            status="done" if ok == len(work) else "partial")

    print(f"\n{'='*60}")
    print(f"Finished: {ok}/{len(work)} video(s) created.")
    print(f"Run folder: outputs/{run_id}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
