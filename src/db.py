"""
src/db.py — SQLite tracker for runs and generations.

Production-extracted verbatim with one path adaptation: DB_PATH anchored
to REPO_ROOT instead of CWD-relative.

Two tables:
  - runs        : one row per pipeline run (run_batch invocation)
  - generations : one row per scenario per run (every Step 1 + Step 2 attempt)

Used for resume-on-failure, parallel-worker coordination, and post-hoc QA scoring.
The `manual_score` and `manual_notes` columns on `generations` are the hook for
the future evaluation layer (filter usable images from a batch before video
generation).
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime
from typing import Iterator

# Repo-root-anchored path (changed from CWD-relative for the new repo)
REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "data" / "alluvi.db"


# ──────────────────────────────────────────────────────────────────────────
# SCHEMA
# ──────────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    timestamp           TEXT NOT NULL,
    plan                TEXT NOT NULL,
    pilot_mode          INTEGER DEFAULT 0,
    total_scenarios     INTEGER DEFAULT 0,
    successful          INTEGER DEFAULT 0,
    failed              INTEGER DEFAULT 0,
    total_cost_usd      REAL DEFAULT 0.0,
    duration_seconds    INTEGER DEFAULT 0,
    notes               TEXT
);

CREATE TABLE IF NOT EXISTS generations (
    gen_id              TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL,
    scenario_id         TEXT NOT NULL,
    plan                TEXT NOT NULL,

    -- Step 1 (PuLID)
    step_1_status       TEXT DEFAULT 'pending',
    step_1_endpoint     TEXT,
    step_1_prompt       TEXT,
    step_1_image_path   TEXT,
    step_1_request_id   TEXT,
    step_1_seed         INTEGER,
    step_1_cost_usd     REAL DEFAULT 0.0,
    step_1_elapsed_s    REAL,
    step_1_error        TEXT,

    -- Step 2 (Qwen-Image-Edit-2511)
    step_2_status       TEXT DEFAULT 'pending',
    step_2_endpoint     TEXT,
    step_2_prompt       TEXT,
    step_2_image_path   TEXT,
    step_2_request_id   TEXT,
    step_2_seed         INTEGER,
    step_2_cost_usd     REAL DEFAULT 0.0,
    step_2_elapsed_s    REAL,
    step_2_error        TEXT,

    -- Overall
    final_status        TEXT DEFAULT 'pending',
    error_message       TEXT,

    -- Manual QA (filled in by human or evaluation layer after run)
    manual_score        INTEGER,
    manual_notes        TEXT,

    created_at          TEXT NOT NULL,
    completed_at        TEXT,

    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_gen_run    ON generations(run_id);
CREATE INDEX IF NOT EXISTS idx_gen_status ON generations(final_status);
CREATE INDEX IF NOT EXISTS idx_gen_scen   ON generations(scenario_id);
"""


# ──────────────────────────────────────────────────────────────────────────
# CONNECTION
# ──────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create the database file and tables if they don't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Context manager yielding a connection with row_factory configured."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


# ──────────────────────────────────────────────────────────────────────────
# RUN HELPERS
# ──────────────────────────────────────────────────────────────────────────

def create_run(run_id: str, plan: str, pilot_mode: bool = False, notes: str = "") -> None:
    init_db()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO runs (run_id, timestamp, plan, pilot_mode, notes)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, datetime.utcnow().isoformat(), plan, int(pilot_mode), notes),
        )
        conn.commit()


def finalize_run(
    run_id: str,
    total_scenarios: int,
    successful: int,
    failed: int,
    total_cost_usd: float,
    duration_seconds: int,
) -> None:
    with connect() as conn:
        conn.execute(
            """
            UPDATE runs SET
                total_scenarios  = ?,
                successful       = ?,
                failed           = ?,
                total_cost_usd   = ?,
                duration_seconds = ?
            WHERE run_id = ?
            """,
            (total_scenarios, successful, failed, total_cost_usd, duration_seconds, run_id),
        )
        conn.commit()


# ──────────────────────────────────────────────────────────────────────────
# GENERATION HELPERS
# ──────────────────────────────────────────────────────────────────────────

def create_generation(gen_id: str, run_id: str, scenario_id: str, plan: str) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO generations (gen_id, run_id, scenario_id, plan, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (gen_id, run_id, scenario_id, plan, datetime.utcnow().isoformat()),
        )
        conn.commit()


def update_step_1(
    gen_id: str,
    *,
    status: str,
    endpoint: str | None = None,
    prompt: str | None = None,
    image_path: str | None = None,
    request_id: str | None = None,
    seed: int | None = None,
    cost_usd: float | None = None,
    elapsed_s: float | None = None,
    error: str | None = None,
) -> None:
    fields, values = [], []
    for col, val in [
        ("step_1_status", status),
        ("step_1_endpoint", endpoint),
        ("step_1_prompt", prompt),
        ("step_1_image_path", image_path),
        ("step_1_request_id", request_id),
        ("step_1_seed", seed),
        ("step_1_cost_usd", cost_usd),
        ("step_1_elapsed_s", elapsed_s),
        ("step_1_error", error),
    ]:
        if val is not None:
            fields.append(f"{col} = ?")
            values.append(val)
    if not fields:
        return
    values.append(gen_id)
    with connect() as conn:
        conn.execute(f"UPDATE generations SET {', '.join(fields)} WHERE gen_id = ?", values)
        conn.commit()


def update_step_2(
    gen_id: str,
    *,
    status: str,
    endpoint: str | None = None,
    prompt: str | None = None,
    image_path: str | None = None,
    request_id: str | None = None,
    seed: int | None = None,
    cost_usd: float | None = None,
    elapsed_s: float | None = None,
    error: str | None = None,
) -> None:
    fields, values = [], []
    for col, val in [
        ("step_2_status", status),
        ("step_2_endpoint", endpoint),
        ("step_2_prompt", prompt),
        ("step_2_image_path", image_path),
        ("step_2_request_id", request_id),
        ("step_2_seed", seed),
        ("step_2_cost_usd", cost_usd),
        ("step_2_elapsed_s", elapsed_s),
        ("step_2_error", error),
    ]:
        if val is not None:
            fields.append(f"{col} = ?")
            values.append(val)
    if not fields:
        return
    values.append(gen_id)
    with connect() as conn:
        conn.execute(f"UPDATE generations SET {', '.join(fields)} WHERE gen_id = ?", values)
        conn.commit()


def finalize_generation(gen_id: str, final_status: str, error_message: str | None = None) -> None:
    with connect() as conn:
        conn.execute(
            """
            UPDATE generations SET
                final_status = ?,
                error_message = ?,
                completed_at = ?
            WHERE gen_id = ?
            """,
            (final_status, error_message, datetime.utcnow().isoformat(), gen_id),
        )
        conn.commit()


def set_manual_score(gen_id: str, score: int, notes: str = "") -> None:
    """Hook for the future evaluation layer — sets manual_score on a generation."""
    with connect() as conn:
        conn.execute(
            "UPDATE generations SET manual_score = ?, manual_notes = ? WHERE gen_id = ?",
            (score, notes, gen_id),
        )
        conn.commit()


# ──────────────────────────────────────────────────────────────────────────
# QUERIES
# ──────────────────────────────────────────────────────────────────────────

def get_run_summary(run_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return dict(row)


def get_generations_for_run(run_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM generations WHERE run_id = ? ORDER BY created_at",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_failed_generations(run_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM generations WHERE run_id = ? AND final_status != 'success'",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]