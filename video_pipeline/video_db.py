"""
video_pipeline/video_db.py — data-access layer for the VIDEO pipeline.

Reuses the SAME Supabase client + conventions as the image pipeline
(supabase_pipeline.supabase_db) — it just adds the two new tables (videos,
media_generations) and the video work-selector. Stage tracking (stage_executions)
and the script LLM call (llm_calls) are written via supabase_db directly, so the
video stages share the exact same audit spine as the image stages.

Public surface:
  get_outputs_needing_video(...)      -> list[dict]   # the work queue (resume)
  get_account_for_persona(persona_id) -> dict|None    # for script gender/language
  upsert_video(...)                   -> dict          # create/replace (1 per output)
  update_video(video_id, **fields)    -> None
  get_video_for_output(output_id)     -> dict|None
  insert_media_generation(...)        -> int
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the existing client + helpers (single source of DB truth).
from supabase_pipeline.supabase_db import client


# ──────────────────────────────────────────────────────────────────────────
# Work selector — finished scene images that still need a video
# ──────────────────────────────────────────────────────────────────────────

def get_outputs_needing_video(
    *,
    persona_ids: Optional[list[int]] = None,
    include_qc_skipped: bool = True,
    limit: Optional[int] = None,
) -> list[dict]:
    """Outputs that are done + QC-acceptable and have NO video row yet."""
    statuses = ["pass"] + (["skipped"] if include_qc_skipped else [])
    q = (
        client().table("outputs").select("*")
        .eq("status", "done").in_("qc_status", statuses)
    )
    if persona_ids:
        q = q.in_("persona_id", persona_ids)
    rows = q.order("id").execute().data or []

    have = {
        v["output_id"]
        for v in (client().table("videos").select("output_id").execute().data or [])
    }
    todo = [r for r in rows if r["id"] not in have]
    return todo[:limit] if limit else todo


def get_account_for_persona(persona_id: int) -> Optional[dict]:
    """persona_id -> the tiktok_accounts row (for gender / language / age)."""
    p = (
        client().table("personas").select("tiktok_account_id")
        .eq("id", persona_id).limit(1).execute().data or [None]
    )[0]
    if not p:
        return None
    return (
        client().table("tiktok_accounts").select("*")
        .eq("id", p["tiktok_account_id"]).limit(1).execute().data or [None]
    )[0]


# ──────────────────────────────────────────────────────────────────────────
# videos  (one per output — UNIQUE output_id; resume-safe)
# ──────────────────────────────────────────────────────────────────────────

def get_video_for_output(output_id: int) -> Optional[dict]:
    res = (
        client().table("videos").select("*")
        .eq("output_id", output_id).limit(1).execute()
    )
    return (res.data or [None])[0]


def upsert_video(
    *,
    output_id: int,
    persona_id: Optional[int] = None,
    run_pk: Optional[int] = None,
    scenario_id: Optional[str] = None,
    dialogue: Optional[str] = None,
    language: Optional[str] = None,
    hook_style: Optional[str] = None,
    scene_mood: Optional[str] = None,
    wan_motion_prompt: Optional[str] = None,
    wan_negative_prompt: Optional[str] = None,
    ref_audio_path: Optional[str] = None,
    ref_text: Optional[str] = None,
    status: str = "running",
) -> dict:
    """Create or replace the video row for an output. Returns the row."""
    row = {
        "output_id": output_id,
        "persona_id": persona_id,
        "run_id": run_pk,
        "scenario_id": scenario_id,
        "dialogue": dialogue,
        "language": language,
        "hook_style": hook_style,
        "scene_mood": scene_mood,
        "wan_motion_prompt": wan_motion_prompt,
        "wan_negative_prompt": wan_negative_prompt,
        "ref_audio_path": ref_audio_path,
        "ref_text": ref_text,
        "status": status,
    }
    res = client().table("videos").upsert(row, on_conflict="output_id").execute()
    return res.data[0]


def update_video(video_id: int, **fields: Any) -> None:
    """Patch arbitrary columns on a video row (audio/video/final paths, status...)."""
    if not fields:
        return
    client().table("videos").update(fields).eq("id", video_id).execute()


# ──────────────────────────────────────────────────────────────────────────
# media_generations  (audio/video analog of image_generations)
# ──────────────────────────────────────────────────────────────────────────

def insert_media_generation(
    *,
    stage_name: str,
    media_type: str,
    video_id: Optional[int] = None,
    stage_execution_id: Optional[int] = None,
    model_name: Optional[str] = None,
    backend: str = "comfyui",
    workflow_template: Optional[str] = None,
    comfyui_server: Optional[str] = None,
    prompt: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    params: Any = None,                 # dict -> jsonb
    input_paths: Any = None,            # list[str] -> jsonb
    output_local_path: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    elapsed_seconds: Optional[float] = None,
    cost_usd: float = 0.0,
) -> int:
    row = {
        "video_id": video_id,
        "stage_execution_id": stage_execution_id,
        "stage_name": stage_name,
        "media_type": media_type,
        "model_name": model_name,
        "backend": backend,
        "workflow_template": workflow_template,
        "comfyui_server": comfyui_server,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "params": params,
        "input_paths": input_paths,
        "output_local_path": output_local_path,
        "duration_seconds": duration_seconds,
        "elapsed_seconds": elapsed_seconds,
        "cost_usd": cost_usd,
    }
    return client().table("media_generations").insert(row).execute().data[0]["id"]


# ──────────────────────────────────────────────────────────────────────────
# Self-test:  python video_pipeline/video_db.py
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[video_db] connecting...")
    todo = get_outputs_needing_video()
    print(f"[video_db] OK — outputs needing a video: {len(todo)}")
    for o in todo[:10]:
        print(f"  - output id={o['id']:<4} persona={o['persona_id']} "
              f"scenario={o['scenario_id']} qc={o.get('qc_status')}")
    existing = client().table("videos").select("id", count="exact").execute()
    print(f"[video_db] existing videos rows: {existing.count or 0}")
