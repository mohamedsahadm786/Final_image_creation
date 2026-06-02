"""
video_pipeline/script_builder.py — Phase-C front layer: turn a finished scene
(persona + scenario) into the spoken line and the Wan animation prompts for one
~5-second clip.

Single source of truth for all CONTENT is brand/alluvi_information.json — it is
loaded live on every call, so editing that file changes behaviour without
touching the rule book. The rule book (prompts/master_prompt_script.md) holds
only the METHODOLOGY + compliance + model constraints.

Pure LLM, no GPU. Test cost ~$0.10.

Public surface:
  build_script(account, scenario) -> dict   # the parsed JSON (see master prompt)

Output keys: dialogue, language, estimated_speech_seconds, hook_style,
scene_mood, wan_motion_prompt, wan_negative_prompt, rationale.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import anthropic
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

BRAND_JSON_PATH = REPO_ROOT / "brand" / "alluvi_information.json"
RULE_BOOK_PATH = Path(__file__).resolve().parent / "prompts" / "master_prompt_script.md"

MODEL = "claude-opus-4-7"
MAX_TOKENS = 1500
SYSTEM_PROMPT_NAME = "master_prompt_script"
SYSTEM_PROMPT_VERSION = "v1"

REQUIRED_KEYS = ["dialogue", "wan_motion_prompt", "wan_negative_prompt"]

# Only these sections of the brand JSON are injected (keeps the call focused on
# what actually drives a script + motion prompt).
_BRAND_SECTIONS = [
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
    """Extract the script-relevant slices of alluvi_information.json (live)."""
    if not BRAND_JSON_PATH.exists():
        raise FileNotFoundError(
            f"brand knowledge not found: {BRAND_JSON_PATH} "
            f"(this is the ONLY content source — it must exist)")
    full = json.loads(BRAND_JSON_PATH.read_text(encoding="utf-8"))
    out = {}
    for section, keys in _BRAND_SECTIONS:
        block = full.get(section, {}) or {}
        picked = {k: block.get(k) for k in keys if block.get(k) is not None}
        if picked:
            out[section] = picked
    return out


def _parse_json(text: str) -> dict:
    """Tolerant JSON extraction (strips fences / prose)."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            raise ValueError(f"no JSON object found in model output:\n{text[:400]}")
        return json.loads(m.group(0))


def build_script(account: dict, scenario: dict) -> dict:
    """Opus -> {dialogue, wan_motion_prompt, wan_negative_prompt, ...}."""
    rule_book = RULE_BOOK_PATH.read_text(encoding="utf-8")
    brand = _load_brand_knowledge()

    persona = {
        "name": account.get("name"),
        "gender": account.get("gender"),
        "country": account.get("country"),
        "language": account.get("language"),
        "age": account.get("age"),
    }
    scene = {
        "scenario_id": scenario.get("id"),
        "category": scenario.get("category"),
        "location": scenario.get("location") or scenario.get("setting"),
        "mood": scenario.get("mood"),
        "activity": scenario.get("activity") or scenario.get("action"),
        "notes": scenario.get("notes"),
        "raw": scenario,  # full scenario, in case it carries extra fields
    }

    user_message = (
        "BRAND KNOWLEDGE (from alluvi_information.json — your only content source):\n"
        + json.dumps(brand, indent=2, ensure_ascii=False)
        + "\n\nPERSONA:\n" + json.dumps(persona, indent=2, ensure_ascii=False)
        + "\n\nSCENE:\n" + json.dumps(scene, indent=2, ensure_ascii=False)
        + "\n\nGenerate the JSON exactly as specified. STRICT JSON only."
    )

    sid = scenario.get("id", "?")
    print(f"[script_builder] Opus {MODEL} for {account.get('name')} / scene {sid}")
    resp = _client().messages.create(
        model=MODEL, max_tokens=MAX_TOKENS,
        system=rule_book,
        messages=[{"role": "user", "content": user_message}],
    )
    raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    parsed = _parse_json(raw)

    missing = [k for k in REQUIRED_KEYS if not parsed.get(k)]
    if missing:
        raise ValueError(f"script JSON missing required keys: {missing}\n{raw[:400]}")

    # soft guards (warn, don't fail — let the operator see the result)
    wc = len(str(parsed["dialogue"]).split())
    if wc > 22:
        print(f"[script_builder]   WARNING dialogue is {wc} words (>~5s target)")
    mw = len(str(parsed["wan_motion_prompt"]).split())
    if mw > 110:
        print(f"[script_builder]   WARNING wan_motion_prompt is {mw} words "
              f"(Wan degrades on long prompts)")

    parsed["_dialogue_word_count"] = wc
    print(f"[script_builder]   done ({wc}-word line, {mw}-word motion prompt)")
    return parsed


# ──────────────────────────────────────────────────────────────────────────
# Standalone test:
#   cd /workspace/alluvi-pipeline && source /workspace/ai-toolkit/venv/bin/activate
#   python video_pipeline/script_builder.py 1        # account id
#   python video_pipeline/script_builder.py 1 dawn_kitchen_routine   # + scenario id
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from supabase_pipeline import supabase_db
    from src import scenario_loader

    accounts = supabase_db.get_all_accounts()
    if not accounts:
        print("no accounts in tiktok_accounts")
        sys.exit(1)
    want = int(sys.argv[1]) if len(sys.argv) > 1 else accounts[0]["id"]
    account = next((a for a in accounts if a["id"] == want), accounts[0])

    scenarios = scenario_loader.load_scenarios()
    if len(sys.argv) > 2:
        scenario = next((s for s in scenarios if s.get("id") == sys.argv[2]), scenarios[0])
    else:
        scenario = scenarios[0]

    print(f"\n=== script test: {account['tiktok_id']} "
          f"({account['gender']}, {account['language']}) / scene {scenario.get('id')} ===\n")
    result = build_script(account, scenario)
    print("\n--- RESULT ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\n--- DIALOGUE ---")
    print(result["dialogue"])
    print("\n--- WAN MOTION ---")
    print(result["wan_motion_prompt"])
    print("\n--- WAN NEGATIVE ---")
    print(result["wan_negative_prompt"])
