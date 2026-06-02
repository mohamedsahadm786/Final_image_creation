"""
video_pipeline/multishot/script_builder_multi.py — multi-shot front layer.

Turns ONE finished scene (persona + scenario) into a coherent N-beat narrative,
where every beat animates the SAME finished realism photo. Live brand JSON is the
only content source, the rule book holds methodology only. Pure LLM, no GPU.

  build_multishot_script(account, scenario, num_shots=2, target_seconds=5) -> dict
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import anthropic
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

BRAND_JSON_PATH = REPO_ROOT / "brand" / "alluvi_information.json"
RULE_BOOK_PATH = Path(__file__).resolve().parent / "prompts" / "master_prompt_multishot.md"

MODEL = "claude-opus-4-7"
MAX_TOKENS = 1500   # base; scaled per shot below
WORDS_PER_SECOND = 3.0   # F5 English pace. Raise if audio lands short, lower if long.
SYSTEM_PROMPT_NAME = "master_prompt_multishot"
SYSTEM_PROMPT_VERSION = "v5"

_BRAND_SECTIONS = [
    ("system_identity", ["brand_name", "product_name", "brand_positioning"]),
    ("brand_personality", ["brand_voice", "core_traits"]),
    ("marketing_language_engine", ["high_performing_phrases", "hook_styles", "conversation_styles"]),
    ("product_knowledge", ["positive_lifestyle_language", "wellness_associations"]),
    ("dialogue_generation_rules", ["dialogue_style", "dialogue_requirements", "example_dialogues"]),
    ("video_generation_preferences", ["camera_motion", "motion_behavior", "visual_style"]),
    ("scene_generation_system", ["scene_moods"]),
    ("ai_generation_priorities", ["highest_priority", "negative_generation_controls"]),
]


def _client() -> anthropic.Anthropic:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY missing in environment / .env")
    return anthropic.Anthropic(api_key=key)


def _load_brand_knowledge() -> dict:
    if not BRAND_JSON_PATH.exists():
        raise FileNotFoundError(f"brand knowledge not found: {BRAND_JSON_PATH}")
    full = json.loads(BRAND_JSON_PATH.read_text(encoding="utf-8"))
    out = {}
    for section, keys in _BRAND_SECTIONS:
        block = full.get(section, {}) or {}
        picked = {k: block.get(k) for k in keys if block.get(k) is not None}
        if picked:
            out[section] = picked
    return out


def _parse_json(text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            raise ValueError(f"no JSON object found in model output:\n{text[:400]}")
        return json.loads(m.group(0))


def build_multishot_script(account: dict, scenario: dict, num_shots: int = 2, target_seconds: int = 10) -> dict:
    rule_book = RULE_BOOK_PATH.read_text(encoding="utf-8")
    brand = _load_brand_knowledge()

    total_target_seconds = num_shots * target_seconds
    target_total_words = round(total_target_seconds * WORDS_PER_SECOND)
    words_per_shot = max(1, round(target_total_words / max(1, num_shots)))

    persona = {k: account.get(k) for k in ("name", "gender", "country", "language", "age")}
    scene = {
        "scenario_id": scenario.get("id"),
        "category": scenario.get("category"),
        "location": scenario.get("location") or scenario.get("setting"),
        "mood": scenario.get("mood"),
        "activity": scenario.get("activity") or scenario.get("action"),
        "notes": scenario.get("notes"),
        "raw": scenario,
    }

    user_message = (
        "BRAND KNOWLEDGE (from alluvi_information.json — your only content source):\n"
        + json.dumps(brand, indent=2, ensure_ascii=False)
        + "\n\nPERSONA:\n" + json.dumps(persona, indent=2, ensure_ascii=False)
        + "\n\nSCENE:\n" + json.dumps(scene, indent=2, ensure_ascii=False)
        + f"\n\nNUM_SHOTS: {num_shots}"
        + f"\n\nTARGET_SECONDS_PER_SHOT: {target_seconds}"
        + f"\n\nTOTAL_TARGET_SECONDS: {total_target_seconds}"
        + f"\n\nTARGET_TOTAL_WORDS: {target_total_words}"
        + f"\n\nWORDS_PER_SHOT: {words_per_shot}"
        + f"\n\nThe COMBINED dialogue across all {num_shots} shots must total about "
        + f"{target_total_words} words (hit this or slightly OVER — never significantly under), "
        + f"about {words_per_shot} words per shot. Name ALLUVI naturally about twice across the "
        + "whole script and reference the product at least once, per the rule book. "
        + f"Generate the JSON exactly as specified, with exactly {num_shots} shots. STRICT JSON only."
    )

    sid = scenario.get("id", "?")
    print(f"[script_builder_multi] Opus {MODEL} for {account.get('name')} / "
          f"scene {sid} / {num_shots} shots / target ~{target_total_words}w (~{total_target_seconds}s)")
    token_budget = min(32000, MAX_TOKENS + num_shots * 900)
    resp = _client().messages.create(
        model=MODEL, max_tokens=token_budget,
        system=rule_book,
        messages=[{"role": "user", "content": user_message}],
    )
    raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    parsed = _parse_json(raw)

    shots = parsed.get("shots")
    if not isinstance(shots, list) or not shots:
        raise ValueError(f"no 'shots' array in output:\n{raw[:400]}")
    if len(shots) != num_shots:
        print(f"[script_builder_multi]   WARNING asked for {num_shots} shots, "
              f"got {len(shots)} — using what was returned")
    total_words = 0
    brand_hits = 0
    for i, sh in enumerate(shots):
        for k in ("dialogue", "wan_motion_prompt", "wan_negative_prompt"):
            if not sh.get(k):
                raise ValueError(f"shot {i} missing '{k}':\n{json.dumps(sh)[:300]}")
        d = str(sh["dialogue"])
        wc = len(d.split()); total_words += wc
        brand_hits += d.upper().count("ALLUVI") + d.upper().count("TIRZEPATIDE")
        mw = len(str(sh["wan_motion_prompt"]).split())
        sh["_dialogue_word_count"] = wc
        flag = "  <-- long" if wc > words_per_shot * 1.6 else ""
        mflag = "  <-- long motion" if mw > 130 else ""
        print(f"[script_builder_multi]   shot {i+1}: {wc}w dialogue{flag}, {mw}w motion{mflag}")
    est = total_words / max(0.1, WORDS_PER_SECOND)
    print(f"[script_builder_multi]   TOTAL {total_words}w (target ~{target_total_words}) "
          f"-> ~{est:.0f}s speech, brand/product mentions: {brand_hits}")
    return parsed


if __name__ == "__main__":
    import sys
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from supabase_pipeline import supabase_db

    acct_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    secs = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    accts = {a["id"]: a for a in supabase_db.get_all_accounts()}
    account = accts.get(acct_id)
    if not account:
        print(f"no account id={acct_id}; available: {sorted(accts)}"); sys.exit(1)

    scenario = {"id": "gym_post_workout_mirror_01", "category": "gym",
                "location": "modern commercial gym", "mood": "quiet confidence",
                "activity": "post-workout mirror moment"}

    print(f"\n=== multishot script test: account={account.get('tiktok_id')} shots={n} ===\n")
    out = build_multishot_script(account, scenario, num_shots=n, target_seconds=secs)
    print("\n--- RESULT ---")
    print(json.dumps(out, indent=2, ensure_ascii=False))
