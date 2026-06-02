"""
supabase_pipeline/phase_a_prompt_builder.py — Phase A persona appearance prompt
generation via Opus 4.7.

Mirrors the shape of src/step_1_prompt_builder.py — same client, same json_utils
validator, same context-loading pattern — but the INPUT is one tiktok_accounts
row (identity factors) instead of a scenario, and the OUTPUT is a persona
identity descriptor set + a text-to-image portrait prompt.

The returned dict's `prompt_descriptors` block carries the exact fields that the
UNCHANGED src/step_1_prompt_builder.py copies verbatim:
  face_descriptor_short, face_descriptor_full,
  identity_lock_minimal, identity_lock_strong, identity_lock_close_up

Pure function: returns a dict. The orchestrator handles DB writes (llm_calls)
and serializing the descriptors into a per-account persona.yaml. Phase A renders
no product and no scene — only the permanent face/body/hair/eyes identity.

Standalone test (no GPU; one Opus call, ~$0.10):
  cd /workspace/alluvi-pipeline
  python supabase_pipeline/phase_a_prompt_builder.py            # first account
  python supabase_pipeline/phase_a_prompt_builder.py 3          # account id=3
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

# Repo root is one level up from supabase_pipeline/ — put it on sys.path so
# `from src.json_utils import ...` resolves (same trick the orchestrator uses).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")

from src.json_utils import validate_json_output  # noqa: E402

CLAUDE_MODEL = "claude-opus-4-7"

SYSTEM_PROMPT_PATH = REPO_ROOT / "prompts" / "master_prompt_phaseA.md"
BRAND_YAML_PATH = REPO_ROOT / "brand" / "brand.yaml"
DO_DONT_MD_PATH = REPO_ROOT / "brand" / "do_dont.md"

# The descriptor fields the downstream step_1 builder copies verbatim. We verify
# Opus actually produced them so a drift fails cleanly here, not three stages later.
REQUIRED_DESCRIPTOR_KEYS = [
    "face_descriptor_short",
    "face_descriptor_full",
    "identity_lock_minimal",
    "identity_lock_strong",
    "identity_lock_close_up",
]

_client: Anthropic | None = None
_static_context_cache: dict[str, str] | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in environment (.env)")
        _client = Anthropic(api_key=api_key)
    return _client


def _load_static_context() -> dict[str, str]:
    """Load + cache brand.yaml + do_dont.md (persona must stay brand-safe)."""
    global _static_context_cache
    if _static_context_cache is not None:
        return _static_context_cache
    paths = {"brand_yaml": BRAND_YAML_PATH, "do_dont_md": DO_DONT_MD_PATH}
    missing = [(k, p) for k, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required context files:\n"
            + "\n".join(f"  - {k}: {p}" for k, p in missing)
        )
    _static_context_cache = {k: p.read_text(encoding="utf-8") for k, p in paths.items()}
    return _static_context_cache


def _parse_json(text: str) -> dict:
    """Defensive parse + require the top-level keys Phase A must produce."""
    return validate_json_output(
        text, required_keys=["prompt_descriptors", "portrait_prompt"]
    )


def build_appearance_prompt(account: dict) -> dict:
    """
    Build the Phase A persona appearance envelope for one tiktok_accounts row.

    Args:
        account: a tiktok_accounts row dict with at least
                 tiktok_id, name, gender, country, age, language.

    Returns:
        dict with keys: identity, hair, eyes, face, prompt_descriptors (the 5
        verbatim fields), anti_features, portrait_prompt, portrait_negative_prompt.
    """
    if not SYSTEM_PROMPT_PATH.exists():
        raise FileNotFoundError(f"missing system prompt: {SYSTEM_PROMPT_PATH}")

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    ctx = _load_static_context()

    tiktok_id = account.get("tiktok_id", "?")
    user_message = "\n".join(
        [
            "=== brand.yaml (persona must fit this brand) ===",
            ctx["brand_yaml"],
            "",
            "=== do_dont.md (compliance) ===",
            ctx["do_dont_md"],
            "",
            "=== ACCOUNT IDENTITY FACTORS (generate the persona for THIS person) ===",
            json.dumps(
                {
                    "tiktok_id": account.get("tiktok_id"),
                    "name": account.get("name"),
                    "gender": account.get("gender"),
                    "country": account.get("country"),
                    "age": account.get("age"),
                    "language": account.get("language"),
                },
                indent=2,
                ensure_ascii=False,
            ),
            "",
            "=== TASK ===",
            "Generate this persona's PERMANENT visual identity per the system prompt.",
            "",
            "HARD REQUIREMENTS:",
            "  - gender_presentation MUST match the account's gender; use the matching",
            "    pronoun (she/her or he/him) in EVERY descriptor and identity-lock line.",
            "  - apparent age tracks the account age but NEVER below 21 (compliance floor).",
            "  - features realistically and respectfully consistent with the country;",
            "    a specific believable person, not an ethnically-ambiguous model.",
            "  - the 5 prompt_descriptors fields are MANDATORY and gender-correct:",
            "    face_descriptor_short, face_descriptor_full, identity_lock_minimal,",
            "    identity_lock_strong, identity_lock_close_up.",
            "  - portrait_prompt: a 60-110 word FRONT-FACING neutral reference headshot,",
            "    plain background, neutral closed-mouth expression, NO product/props/text,",
            "    with the photoreal open/close anchors. Gender-correct pronoun.",
            "  - body proportions healthy and natural — never emaciated, never exaggerated.",
            "",
            "Output JSON only. No preamble. No markdown fences.",
        ]
    )

    print(f"[phase_a_prompt_builder] Phase A -> Opus 4.7 for account {tiktok_id}")
    response = _get_client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    output = _parse_json(response.content[0].text)

    # Secondary check: the 5 verbatim descriptor fields must be present + non-empty.
    descriptors = output.get("prompt_descriptors") or {}
    missing = [k for k in REQUIRED_DESCRIPTOR_KEYS if not descriptors.get(k)]
    if missing:
        from src.json_utils import JSONSanityError
        raise JSONSanityError(
            f"Phase A output missing/empty prompt_descriptors fields: {missing}"
        )

    print(f"[phase_a_prompt_builder]   done for {tiktok_id}")
    return output


# ──────────────────────────────────────────────────────────────────────────
# Standalone test — no GPU, one Opus call. Pulls a real account from Supabase.
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import supabase_db  # sibling module (supabase_pipeline/ is on sys.path[0])

    accounts = supabase_db.get_all_accounts()
    if not accounts:
        print("[phase_a_prompt_builder] no accounts in tiktok_accounts — add some first")
        sys.exit(1)

    if len(sys.argv) > 1:
        want = int(sys.argv[1])
        account = next((a for a in accounts if a["id"] == want), None)
        if account is None:
            print(f"[phase_a_prompt_builder] no account with id={want}")
            sys.exit(1)
    else:
        account = accounts[0]

    print(f"\n=== ACCOUNT id={account['id']} {account['tiktok_id']} "
          f"({account['gender']}, {account['country']}, age {account['age']}, "
          f"{account['language']}) ===\n")

    result = build_appearance_prompt(account)

    pd = result["prompt_descriptors"]
    print("--- face_descriptor_short ---")
    print(pd["face_descriptor_short"])
    print("\n--- face_descriptor_full ---")
    print(pd["face_descriptor_full"])
    print("\n--- identity_lock_strong ---")
    print(pd["identity_lock_strong"])
    print("\n--- portrait_prompt ---")
    print(result["portrait_prompt"])
    print("\n--- full JSON (saved nowhere — this is just the test) ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))
