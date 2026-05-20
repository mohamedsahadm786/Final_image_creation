"""
src/step_2_prompt_builder.py — Qwen-tuned Step 2 prompt generation via Opus 4.7.

v6 — Research-tuned tagged-section structure (180-240 words, hard ceiling 280).

Key changes from v5 (in this version):
  - 6-section tagged structure: EDIT / PRODUCT / PRESERVE / ANATOMY / UNIQUENESS / LIGHTING
  - PRODUCT section quotes the exact product text strings verbatim (research:
    apiyi.com 23-test study showed quoting text raises Qwen rendering accuracy
    from 65% to 96%)
  - PRESERVE section is categorical only — no persona/outfit/scene re-description
    (the model already sees the first image; re-description dilutes attention
    AND creates contradictions when Stage 1 differs from scenario intent)
  - NEW ANATOMY clause — natural human anatomy without counting fingers.
    The old "five fingers per hand (one thumb plus four others) — fingers
    occluded still fully exist" caused Qwen to render visible extras around
    the product. The new clause explicitly forbids this.
  - Reduced word budget from 380-450 to 180-240 (target), hard ceiling 280.
    Research (apiyi, FAL, Replicate) consistently shows shorter prompts win
    for image-edit tasks; the 380-450 target was a hypothesis that didn't hold up.

Backward compatibility (intentionally preserved):
  - Function signature: build_step_2_prompt(scenario, step_1_output) -> dict
  - Returned dict's `step_2_image_prompt` key (used by step_2_qwen_edit.generate)
  - Returned dict's `fal_qwen_params.image_size` (consumed by step_2_qwen_edit)
  - Module path / import path unchanged
  - Static context loader unchanged
  - Anthropic client init unchanged
  - JSON parse defensive guard unchanged
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
QWEN_SYSTEM_PROMPT_PATH = REPO_ROOT / "prompts" / "master_prompt_step2_qwen.md"

PERSONA_YAML_PATH = REPO_ROOT / "assets" / "persona.yaml"
PRODUCT_YAML_PATH = REPO_ROOT / "assets" / "product.yaml"
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
    """Load and cache the four static context files."""
    global _static_context_cache
    if _static_context_cache is not None:
        return _static_context_cache

    paths = {
        "persona_yaml": PERSONA_YAML_PATH,
        "product_yaml": PRODUCT_YAML_PATH,
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
    Defensive JSON parse for Opus output. Uses shared validator in src/json_utils.py.
    Required-key check: step_2_image_prompt must be present.
    """
    from src.json_utils import validate_json_output

    return validate_json_output(
        text,
        required_keys=["step_2_image_prompt"],
    )


def build_step_2_prompt(scenario: dict, step_1_output: dict) -> dict:
    """
    Build the Qwen-tuned Step 2 prompt envelope.

    v6: produces a 6-section tagged prompt (EDIT / PRODUCT / PRESERVE / ANATOMY /
    UNIQUENESS / LIGHTING) at 180-240 words. Quotes product text verbatim per
    apiyi.com 23-test Qwen study (65%→96% text rendering accuracy improvement).

    Args:
        scenario: parsed scenarios.yaml entry for this scenario
        step_1_output: parsed Step 1 prompt envelope from this run

    Returns:
        dict with keys: step_2_image_prompt, word_count, structure_breakdown
        (6-section breakdown), fal_qwen_params, image_inputs_required, compliance_check.
        Schema is backward-compatible with step_2_qwen_edit.generate() consumer.
    """
    if not QWEN_SYSTEM_PROMPT_PATH.exists():
        raise FileNotFoundError(f"missing system prompt: {QWEN_SYSTEM_PROMPT_PATH}")

    system_prompt = QWEN_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    ctx = _load_static_context()

    user_message = "\n".join(
        [
            "=== product.yaml (QUOTE TEXT VERBATIM IN PRODUCT SECTION) ===",
            ctx["product_yaml"],
            "",
            "=== do_dont.md (compliance) ===",
            ctx["do_dont_md"],
            "",
            "=== ORIGINAL SCENARIO ===",
            json.dumps(scenario, indent=2),
            "",
            "=== STEP 1 OUTPUT (use lighting language from sentence 4 in your LIGHTING section) ===",
            json.dumps(step_1_output, indent=2),
            "",
            "=== TASK ===",
            "Build the Qwen-tuned Step 2 prompt envelope using the v6 tagged-section",
            "structure (EDIT / PRODUCT / PRESERVE FROM FIRST IMAGE / ANATOMY / UNIQUENESS",
            "/ LIGHTING). Match the calibration examples in the system prompt closely.",
            "",
            "Word budget: 180-240 words target, HARD CEILING 280. Do NOT exceed.",
            "  This is much shorter than v5's 380-450. Research (apiyi 23-test Qwen",
            "  study, FAL's developer guide, Replicate guidance) consistently shows",
            "  Qwen-Image-Edit responds best to brief, structured prompts. The 449-word",
            "  v5 prompts were producing extra-finger artifacts, garbled product text,",
            "  and persona drift from attention dilution.",
            "",
            "ALL SIX SECTIONS ARE REQUIRED in this order:",
            "",
            "1. EDIT — the single change to make. 25-55 words.",
            "   - Use 'the person from the first image' and 'the product from the second image'.",
            "   - For archetype=placed_on_surface: 'Place the Alluvi product (from the second",
            "     image) on <surface from scenario.grip_or_placement> between <props>... The",
            "     persona's pose stays exactly as in the first image.'",
            "   - For archetype=held_*: 'She is now holding the Alluvi product (from the second",
            "     image) in her <hand> hand at <position> — at <position> specifically, not",
            "     <2-3 negative exclusions max>... Her body posture and <holding-arm> may shift",
            "     naturally for the holding pose; everything else stays from the first image.'",
            "   - For archetype=flat_lay: 'Compose a flat-lay arrangement with the Alluvi",
            "     product (from the second image) centered, surrounded by <props from scene>.",
            "     Shot from directly above.'",
            "",
            "2. PRODUCT — quoted-text packaging spec. ~85 words. SAME across all scenarios.",
            "   Use this exact text (preserve quote marks — they are the highest-leverage",
            "   fidelity tool per the apiyi study):",
            "   --------",
            '   PRODUCT (preserve exactly, from the second image): A horizontal rectangular',
            '   white cardboard box with the text "TIRZEPATIDE", "DUAL AGONIST OF GLP-1,',
            '   GIP RECEPTORS", "ALLUVI", "HEALTHCARE", "40mg" on the front face. Flowing',
            '   blue wave-mesh gradient diagonally across the lower front face. Circular',
            '   green "GOOD MANUFACTURING PRACTICE CERTIFIED" seal in the center. White',
            '   base color. Approximately 7 inches wide by 3 inches tall. The printed',
            '   design rotates with the box as one coherent surface — never reflowed,',
            '   redesigned, mirrored, or text-reversed.',
            "   --------",
            "",
            "3. PRESERVE FROM FIRST IMAGE — categorical only. ~20-30 words.",
            "   - List CATEGORIES: face, hair, skin, body, outfit, jewelry, pose, scene, lighting.",
            "   - Add a ONE-PHRASE scene cue in parens (e.g., 'boutique hotel bedroom').",
            "   - For held_* archetypes: replace 'pose' with 'the left/right hand position",
            "     and legs' (or whichever non-holding limbs are preserved).",
            "   - DO NOT describe the persona's appearance (face, outfit, hair) in detail.",
            "     The model already sees this in image #1. Re-describing creates",
            "     contradictions and dilutes attention.",
            "",
            "4. ANATOMY — the new clause (REQUIRED VERBATIM):",
            "   --------",
            "   ANATOMY: natural human anatomy — two arms, two hands, two legs. Fingers",
            "   that grip or pass behind the product stay HIDDEN behind it — do NOT render",
            "   additional visible fingers around the product to 'complete' the hand. The",
            "   hand should read like a real photograph: some fingers visible, some",
            "   naturally occluded. No extra limbs.",
            "   --------",
            "   DO NOT count fingers. The old 'five fingers per hand' clause caused Qwen",
            "   to render extras around the product. The new clause explicitly forbids this.",
            "",
            "5. UNIQUENESS — single sentence:",
            "   - 'Exactly ONE Alluvi product is visible in the scene.'",
            "   - For mirror scenarios add: '(A mirror reflection of the held product counts",
            "     as the same product, not a duplicate.)'",
            "",
            "6. LIGHTING — echo Step 1's sentence 4 lighting language. ~30-50 words.",
            "   - Quote or near-quote the lighting description from Step 1.",
            "   - End with: 'Apply this directional light to the product's white surface",
            "     as illumination — do not tint the white toward the scene's color cast.'",
            "",
            "The step_2_image_prompt should be ONE STRING with the section labels (EDIT:,",
            "PRODUCT:, etc.) inline followed by their content. Use newline-newline (\\n\\n)",
            "between sections for readability.",
            "",
            "BANNED in this v6 prompt:",
            "  - Persona/outfit/scene re-description in detail (categorical only)",
            "  - Counting fingers (anywhere — 'five fingers', 'four fingers plus thumb')",
            "  - 'fingers occluded still fully exist' or any variant",
            "  - 4+ negative exclusions stacked on a single positive (keep to 2-3 max)",
            "  - Repeated 'keep X unchanged' anchors (one PRESERVE section is enough)",
            "  - Describing product packaging text WITHOUT quote marks",
            "  - 'Match the lighting' without echoing Step 1's lighting language",
        ]
    )

    scenario_id = scenario.get("id", "?")
    print(f"[step_2_prompt_builder] Step 2 (Qwen v6 tagged) -> Opus 4.7 for scenario {scenario_id}")
    response = _get_client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    output = _parse_json(response.content[0].text)

    wc = output.get("word_count", 0)
    print(f"[step_2_prompt_builder]   Step 2 done: word_count={wc}")
    return output