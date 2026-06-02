"""
supabase_pipeline/supabase_db.py — Supabase data-access layer for the Alluvi
IMAGE pipeline. Replaces the SQLite src/db.py.

Talks to the 8-table schema (v2.1):
  tiktok_accounts, personas, pipeline_runs, outputs,
  stage_executions, llm_calls, image_generations, qc_checks

Design:
  - One lazy module-level client built from SUPABASE_URL + SUPABASE_SECRET_KEY,
    read from the EXISTING repo-root .env (same file that holds ANTHROPIC_API_KEY).
  - Small, single-purpose functions. Inserts return the new row's integer id.
  - jsonb columns (parsed_json, issues, scores, raw_result, input_image_paths)
    accept Python dicts/lists directly.
  - Read helpers implement the orchestrator's account-selection + per-persona
    scenario-resume logic.

Credentials (server-side ONLY — never expose the secret key to a browser):
  SUPABASE_URL=https://<project-ref>.supabase.co
  SUPABASE_SECRET_KEY=sb_secret_...

Self-test (verifies connectivity + counts your tiktok_accounts rows):
  cd /workspace/alluvi-pipeline
  python supabase_pipeline/supabase_db.py
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from supabase import create_client, Client

# Load the repo-root .env explicitly (this file lives one level below the root).
REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

_client: Optional[Client] = None


# ──────────────────────────────────────────────────────────────────────────
# Connection
# ──────────────────────────────────────────────────────────────────────────

def client() -> Client:
    """Lazy-init the Supabase client from env. Raises a clear error if unset."""
    global _client
    if _client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SECRET_KEY")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and/or SUPABASE_SECRET_KEY missing. Add them to "
                f"{REPO_ROOT / '.env'} (use the sb_secret_... key, server-side only)."
            )
        _client = create_client(url, key)
    return _client


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ping() -> int:
    """Connectivity self-test. Returns the tiktok_accounts row count."""
    res = client().table("tiktok_accounts").select("id", count="exact").execute()
    return res.count or 0


# ──────────────────────────────────────────────────────────────────────────
# tiktok_accounts  (read-only — humans fill this table)
# ──────────────────────────────────────────────────────────────────────────

def get_all_accounts() -> list[dict]:
    return client().table("tiktok_accounts").select("*").order("id").execute().data or []


def get_accounts_by_tiktok_ids(tiktok_ids: list[str]) -> list[dict]:
    """Resolve a list of TikTok handles (the human key) to full account rows."""
    if not tiktok_ids:
        return []
    return (
        client().table("tiktok_accounts").select("*")
        .in_("tiktok_id", tiktok_ids).order("id").execute().data or []
    )


def get_accounts_without_persona() -> list[dict]:
    """Accounts that have NO personas row yet (the '--all-accounts OFF' set)."""
    accounts = get_all_accounts()
    persona_rows = client().table("personas").select("tiktok_account_id").execute().data or []
    have = {r["tiktok_account_id"] for r in persona_rows}
    return [a for a in accounts if a["id"] not in have]


# ──────────────────────────────────────────────────────────────────────────
# personas  (Phase A — one per account; idempotent via UNIQUE tiktok_account_id)
# ──────────────────────────────────────────────────────────────────────────

def get_persona_for_account(tiktok_account_id: int) -> Optional[dict]:
    res = (
        client().table("personas").select("*")
        .eq("tiktok_account_id", tiktok_account_id).limit(1).execute()
    )
    return (res.data or [None])[0]


def upsert_persona(
    *,
    tiktok_account_id: int,
    appearance_spec: Optional[str] = None,
    prompt_used: Optional[str] = None,
    portrait_local_path: Optional[str] = None,
    status: str = "done",
) -> dict:
    """Insert or update the persona for an account. Returns the persona row."""
    row = {
        "tiktok_account_id": tiktok_account_id,
        "appearance_spec": appearance_spec,
        "prompt_used": prompt_used,
        "portrait_local_path": portrait_local_path,
        "status": status,
    }
    res = client().table("personas").upsert(row, on_conflict="tiktok_account_id").execute()
    return res.data[0]


# ──────────────────────────────────────────────────────────────────────────
# pipeline_runs  (batch header)
# ──────────────────────────────────────────────────────────────────────────

def create_run(
    *,
    run_id: str,
    flow_name: str,
    scenario_filter: Optional[str] = None,
    total_scenarios: int = 0,
    notes: Optional[str] = None,
) -> int:
    row = {
        "run_id": run_id,
        "flow_name": flow_name,
        "scenario_filter": scenario_filter,
        "total_scenarios": total_scenarios,
        "status": "running",
        "notes": notes,
    }
    return client().table("pipeline_runs").insert(row).execute().data[0]["id"]


def finalize_run(run_pk: int, *, status: str, total_cost_usd: float = 0.0) -> None:
    client().table("pipeline_runs").update({
        "status": status,
        "total_cost_usd": total_cost_usd,
        "finished_at": _now(),
    }).eq("id", run_pk).execute()


# ──────────────────────────────────────────────────────────────────────────
# stage_executions  (the spine — one row per stage per image)
# ──────────────────────────────────────────────────────────────────────────

def start_stage(
    *,
    stage_name: str,
    run_pk: Optional[int] = None,
    persona_id: Optional[int] = None,
    output_id: Optional[int] = None,
) -> int:
    row = {
        "run_id": run_pk,
        "persona_id": persona_id,
        "output_id": output_id,
        "stage_name": stage_name,
        "status": "running",
    }
    return client().table("stage_executions").insert(row).execute().data[0]["id"]


def finish_stage(
    stage_id: int,
    *,
    status: str,
    elapsed_seconds: Optional[float] = None,
    error_message: Optional[str] = None,
) -> None:
    client().table("stage_executions").update({
        "status": status,
        "elapsed_seconds": elapsed_seconds,
        "error_message": error_message,
        "finished_at": _now(),
    }).eq("id", stage_id).execute()


# ──────────────────────────────────────────────────────────────────────────
# outputs  (Phase B canonical image — one per persona x scenario; resumable)
# ──────────────────────────────────────────────────────────────────────────

def get_done_scenario_ids(persona_id: int) -> set[str]:
    """scenario_ids that already have an outputs row for this persona (resume)."""
    rows = (
        client().table("outputs").select("scenario_id")
        .eq("persona_id", persona_id).execute().data or []
    )
    return {r["scenario_id"] for r in rows}


def get_output(persona_id: int, scenario_id: str) -> Optional[dict]:
    res = (
        client().table("outputs").select("*")
        .eq("persona_id", persona_id).eq("scenario_id", scenario_id)
        .limit(1).execute()
    )
    return (res.data or [None])[0]


def upsert_output(
    *,
    persona_id: int,
    scenario_id: str,
    scenario_title: Optional[str] = None,
    run_pk: Optional[int] = None,
    final_image_local_path: Optional[str] = None,
    qc_status: Optional[str] = None,
    qc_reason: Optional[str] = None,
    attempts: int = 1,
    status: str = "done",
) -> dict:
    row = {
        "persona_id": persona_id,
        "scenario_id": scenario_id,
        "scenario_title": scenario_title,
        "run_id": run_pk,
        "final_image_local_path": final_image_local_path,
        "qc_status": qc_status,
        "qc_reason": qc_reason,
        "attempts": attempts,
        "status": status,
    }
    res = client().table("outputs").upsert(row, on_conflict="persona_id,scenario_id").execute()
    return res.data[0]


# ──────────────────────────────────────────────────────────────────────────
# llm_calls  (every Anthropic call — prompt builders + QC)
# ──────────────────────────────────────────────────────────────────────────

def insert_llm_call(
    *,
    purpose: str,
    model: str,
    stage_execution_id: Optional[int] = None,
    system_prompt_name: Optional[str] = None,
    system_prompt_version: Optional[str] = None,
    user_message: Optional[str] = None,
    raw_response: Optional[str] = None,
    parsed_json: Any = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    cost_usd: float = 0.0,
    latency_ms: Optional[int] = None,
) -> int:
    row = {
        "stage_execution_id": stage_execution_id,
        "purpose": purpose,
        "model": model,
        "system_prompt_name": system_prompt_name,
        "system_prompt_version": system_prompt_version,
        "user_message": user_message,
        "raw_response": raw_response,
        "parsed_json": parsed_json,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "latency_ms": latency_ms,
    }
    return client().table("llm_calls").insert(row).execute().data[0]["id"]


# ──────────────────────────────────────────────────────────────────────────
# image_generations  (every diffusion call — maps onto the *_request.json audit)
# ──────────────────────────────────────────────────────────────────────────

def insert_image_generation(
    *,
    stage_name: str,
    stage_execution_id: Optional[int] = None,
    model_name: Optional[str] = None,
    endpoint: Optional[str] = None,
    backend: Optional[str] = None,
    workflow_template: Optional[str] = None,
    prompt: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    input_image_paths: Any = None,   # list[str] -> jsonb
    num_inference_steps: Optional[int] = None,
    cfg: Optional[float] = None,
    guidance: Optional[float] = None,
    seed: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    target_size: Optional[int] = None,
    id_weight: Optional[float] = None,
    max_sequence_length: Optional[int] = None,
    output_local_path: Optional[str] = None,
    drive_file_id: Optional[str] = None,
    drive_url: Optional[str] = None,
    request_id: Optional[str] = None,
    elapsed_seconds: Optional[float] = None,
    cost_usd: float = 0.0,
    attempt_number: int = 1,
    comfyui_server: Optional[str] = None,
) -> int:
    row = {
        "stage_execution_id": stage_execution_id,
        "stage_name": stage_name,
        "model_name": model_name,
        "endpoint": endpoint,
        "backend": backend,
        "workflow_template": workflow_template,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "input_image_paths": input_image_paths,
        "num_inference_steps": num_inference_steps,
        "cfg": cfg,
        "guidance": guidance,
        "seed": seed,
        "width": width,
        "height": height,
        "target_size": target_size,
        "id_weight": id_weight,
        "max_sequence_length": max_sequence_length,
        "output_local_path": output_local_path,
        "drive_file_id": drive_file_id,
        "drive_url": drive_url,
        "request_id": request_id,
        "elapsed_seconds": elapsed_seconds,
        "cost_usd": cost_usd,
        "attempt_number": attempt_number,
        "comfyui_server": comfyui_server,
    }
    return client().table("image_generations").insert(row).execute().data[0]["id"]


# ──────────────────────────────────────────────────────────────────────────
# qc_checks  (every QC attempt — not just the final one)
# ──────────────────────────────────────────────────────────────────────────

def insert_qc_check(
    *,
    qc_model: str,
    passed: Optional[bool],
    stage_execution_id: Optional[int] = None,
    output_id: Optional[int] = None,
    image_generation_id: Optional[int] = None,
    attempt_number: int = 1,
    qc_reason: Optional[str] = None,
    issues: Any = None,             # list[str] -> jsonb
    limb_description: Optional[str] = None,
    scores: Any = None,             # dict -> jsonb
    avoid_line: Optional[str] = None,
    image_evaluated_path: Optional[str] = None,
    raw_result: Any = None,         # dict -> jsonb
) -> int:
    row = {
        "stage_execution_id": stage_execution_id,
        "output_id": output_id,
        "image_generation_id": image_generation_id,
        "attempt_number": attempt_number,
        "qc_model": qc_model,
        "passed": passed,
        "qc_reason": qc_reason,
        "issues": issues,
        "limb_description": limb_description,
        "scores": scores,
        "avoid_line": avoid_line,
        "image_evaluated_path": image_evaluated_path,
        "raw_result": raw_result,
    }
    return client().table("qc_checks").insert(row).execute().data[0]["id"]


# ──────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[supabase_db] connecting...")
    n = ping()
    print(f"[supabase_db] OK — tiktok_accounts has {n} row(s).")
    if n:
        for a in get_all_accounts():
            print(f"  - id={a['id']:<3} {a['tiktok_id']:<18} "
                  f"{a['gender']:<7} {a['country']:<16} age={a['age']} {a['language']}")
    print(f"[supabase_db] accounts without a persona yet: "
          f"{len(get_accounts_without_persona())}")
