"""
src/step_1_prompt_builder.py — Stage 1 PuLID prompt generation via Opus 4.7.

Mirrors the shape of src/step_2_prompt_builder.py — same client, same JSON
validator, same context-loading pattern. The differences are Stage-1 specific:
  - System prompt file: prompts/master_prompt_step1.md
  - Function signature: build_step_1_prompt(scenario) — no step_1_output input
  - Static context: persona.yaml + brand.yaml + do_dont.md (NO product.yaml,
    Stage 1 does not render the product)
  - Required JSON keys: ["step_1_image_prompt"]

Production rules carried over from master_prompt_step1.md (post-edits):
  - Word budget: 200-250 standard, 240-280 close-up. Hard ceiling 290.
  - fal_pulid_params.max_sequence_length MUST be "512" (NOT "256") — at "256"
    the T5XXL encoder truncates at ~210 effective words and the sentence-5
    camera anchor + identity-lock line are silently dropped.
  - fal_pulid_params.negative_prompt MUST be present in every envelope.
  - Persona descriptors and identity_lock lines from persona.yaml must be
    copied VERBATIM, never paraphrased.
  - Early + late photoreal anchors are mandatory (anti-cartoonish drift).

This file REPLACES a previously-corrupted version that contained json_utils.py
content under the wrong filename. Without this fix, run.py would crash on
Stage 1 with: AttributeError: module 'step_1_prompt_builder' has no attribute
'build_step_1_prompt'.
"""

import os
import json
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

CLAUDE_MODEL = "claude-opus-4-7"

# Project paths — repo root is two levels up from this file
REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT_PATH = REPO_ROOT / "prompts" / "master_prompt_step1.md"

PERSONA_YAML_PATH = REPO_ROOT / "assets" / "persona.yaml"
BRAND_YAML_PATH = REPO_ROOT / "brand" / "brand.yaml"
DO_DONT_MD_PATH = REPO_ROOT / "brand" / "do_dont.md"

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
    """Load and cache the three static context files (no product.yaml for Stage 1)."""
    global _static_context_cache
    if _static_context_cache is not None:
        return _static_context_cache

    paths = {
        "persona_yaml": PERSONA_YAML_PATH,
        "brand_yaml": BRAND_YAML_PATH,
        "do_dont_md": DO_DONT_MD_PATH,
    }
    missing = [(k, p) for k, p in paths.items() if not p.exists()]
    if missing:
        msg = "Missing required context files:\n" + "\n".join(
            f"  - {k}: {p}" for k, p in missing
        )
        raise FileNotFoundError(msg)

    _static_context_cache = {k: p.read_text(encoding="utf-8") for k, p in paths.items()}
    return _static_context_cache


def _parse_json(text: str) -> dict:
    """
    Defensive JSON parse for Opus output.

    Uses the shared validator in src/json_utils.py which handles:
      - markdown code fences
      - leading/trailing prose
      - trailing commas

    Required-key check ensures `step_1_image_prompt` is present and non-empty.
    Opus almost always produces this correctly, but a sanity guard keeps
    failures clean if it ever drifts.
    """
    from src.json_utils import validate_json_output

    return validate_json_output(
        text,
        required_keys=["step_1_image_prompt"],
    )


def build_step_1_prompt(scenario: dict) -> dict:
    """
    Build the Step 1 PuLID prompt envelope.

    Receives the scenario dict only — Stage 1 doesn't depend on any previous
    stage output. Calls Opus 4.7 with master_prompt_step1.md and returns
    the parsed JSON envelope.

    Args:
        scenario: parsed scenarios.yaml entry for this scenario

    Returns:
        dict with keys: step_1_image_prompt (str), word_count (int),
        structure_breakdown, fal_pulid_params (with max_sequence_length="512"
        and negative_prompt), step_2_brief (or product_slot), compliance_check,
        id_weight_recommendation.
    """
    if not SYSTEM_PROMPT_PATH.exists():
        raise FileNotFoundError(f"missing system prompt: {SYSTEM_PROMPT_PATH}")

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    ctx = _load_static_context()

    user_message = "\n".join(
        [
            "=== persona.yaml (use prompt_descriptors VERBATIM — never paraphrase) ===",
            ctx["persona_yaml"],
            "",
            "=== brand.yaml ===",
            ctx["brand_yaml"],
            "",
            "=== do_dont.md (compliance) ===",
            ctx["do_dont_md"],
            "",
            "=== SCENARIO ===",
            json.dumps(scenario, indent=2),
            "",
            "=== TASK ===",
            "Build the Step 1 prompt envelope per the system prompt above.",
            "",
            "WORD BUDGET (STRICT):",
            "  step_1_image_prompt: 200-250 words for standard framing,",
            "  240-280 for close-up scenarios (framing field contains 'close-up').",
            "  Hard ceiling 290 words. Every word must earn its place. Do not pad.",
            "",
            "PERSONA DESCRIPTOR (VERBATIM):",
            "  Copy face_descriptor_short OR face_descriptor_full from persona.yaml",
            "  exactly. Do not paraphrase. Choose:",
            "    - face_descriptor_short + identity_lock_strong for medium framing",
            "    - face_descriptor_full + identity_lock_close_up for close-up framing",
            "    - face_descriptor_short + identity_lock_minimal for full-body wide",
            "",
            "PHOTOREAL ANCHORS (MANDATORY):",
            "  Early anchor in sentence 1 (immediately after persona descriptor):",
            "    'captured in a candid amateur smartphone snapshot with natural skin",
            "    texture and visible pores. She is wearing...'",
            "  Late anchor in sentence 5 (before identity-lock line):",
            "    'Shot on iPhone 15 Pro [camera variant]. Real photograph, not",
            "    AI-generated, no model pose, candid moment.'",
            "",
            "fal_pulid_params (HARD REQUIREMENTS — ALL MUST BE PRESENT):",
            "  - max_sequence_length: \"512\" (NEVER \"256\" — \"256\" truncates the",
            "    sentence-5 camera anchor and identity-lock line on every call)",
            "  - negative_prompt: anti-defect comma-separated keyword list including",
            "    'plastic skin, airbrushed skin, extra limbs, six fingers, fused",
            "    fingers, deformed hands, distorted face, AI-generated look,",
            "    illustration, 3D render, watermark, text, signature'",
            "  - id_weight: 1.0 (fal API cap) for persona shots, 0.5 for flat-lays",
            "  - true_cfg: 1.5 medium, 1.7 close-up, 1.2 full-body wide",
            "  - guidance_scale: 3.5 default",
            "  - num_inference_steps: 30",
            "  - image_size: {width: 768, height: 1344}",
            "  - num_images: 1, output_format: \"jpeg\", enable_safety_checker: true",
            "",
            "NO PRODUCT in Step 1. Never mention 'Alluvi', 'Tirzepatide', 'the box',",
            "  'the product', 'the packaging'. The hand reserved for the product",
            "  must be described as 'currently empty' with a relaxed open position.",
            "",
            "Output JSON only. No preamble. No markdown fences.",
        ]
    )

    scenario_id = scenario.get("id", "?")
    print(f"[step_1_prompt_builder] Step 1 -> Opus 4.7 for scenario {scenario_id}")
    response = _get_client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    output = _parse_json(response.content[0].text)

    wc = output.get("word_count", 0)
    print(f"[step_1_prompt_builder]   Step 1 done: word_count={wc}")
    return output