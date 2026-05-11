"""
ollama_flow/src/step_2_prompt_builder_ollama.py — Step 2 prompt via Ollama.

Mirrors the public API of the production Claude builder:
  build_step_2_prompt(scenario, step_1_output) -> dict

Reads master prompts and static context from the PARENT repo:
  ../../prompts/master_prompt_step2_qwen.md
  ../../assets/persona.yaml
  ../../assets/product.yaml
  ../../brand/brand.yaml
  ../../brand/do_dont.md

Same TASK section content as the production Claude builder (BREVITY,
rigid-rotation, two-leg+occlusion, single-product, white base preservation).

Quality expectations for Step 2 are LOWER than Step 1 because Step 2 has
many more required clauses to carry. A 7B model will almost certainly drop
at least one of these per output. This builder logs WARNINGS for each
missing clause so you can spot which scenarios got weak prompts.
"""

import json
from pathlib import Path

from . import ollama_client


PARENT_REPO_ROOT = Path(__file__).resolve().parents[2]

QWEN_SYSTEM_PROMPT_PATH = PARENT_REPO_ROOT / "prompts" / "master_prompt_step2_qwen.md"
PERSONA_YAML_PATH = PARENT_REPO_ROOT / "assets" / "persona.yaml"
PRODUCT_YAML_PATH = PARENT_REPO_ROOT / "assets" / "product.yaml"
BRAND_YAML_PATH = PARENT_REPO_ROOT / "brand" / "brand.yaml"
DO_DONT_MD_PATH = PARENT_REPO_ROOT / "brand" / "do_dont.md"


_static_context_cache: dict[str, str] | None = None


OLLAMA_OPTIONS = {
    "temperature": 0.3,
    "top_p": 0.9,
    "num_predict": 4096,
    "repeat_penalty": 1.1,
}


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


def _warn_if_weak(output: dict, scenario_id: str) -> None:
    """Emit visible warnings when Ollama Step 2 output looks weak.

    Note: validity of `step_2_image_prompt` is already enforced by the
    JSONSanityError raised in ollama_client.generate_json. This function
    catches softer issues — word count drift, missing clauses, etc.
    """
    warnings = []

    prompt_text = output["step_2_image_prompt"]
    word_count = len(prompt_text.split())
    # Accept either schema variant — qwen2.5:7b sometimes emits
    # `step_2_image_prompt_word_count` instead of the spec's `word_count`.
    declared = (
        output.get("word_count")
        or output.get("step_2_image_prompt_word_count")
    )
    lowered = prompt_text.lower()

    # Word count check (target 320-410, ceiling 430)
    if word_count < 220:
        warnings.append(
            f"step_2_image_prompt is too short ({word_count} words, target 320-410)"
        )
    elif word_count > 480:
        warnings.append(
            f"step_2_image_prompt is over-long ({word_count} words, "
            f"target 320-410, hard ceiling 430)"
        )
    if declared is not None and abs(declared - word_count) > 20:
        warnings.append(
            f"declared word_count ({declared}) disagrees with actual ({word_count})"
        )

    # Positional reference syntax check
    if "first image" not in lowered or "second image" not in lowered:
        warnings.append(
            "missing positional reference syntax ('the person from the first "
            "image' / 'the product from the second image') — Qwen variant requires it"
        )

    # "Keep X unchanged" anchor check
    if "unchanged" not in lowered:
        warnings.append(
            "no 'keep X unchanged' anchors found — Sentence 1 should thread "
            "'keep her face unchanged' / 'keep her outfit unchanged' style anchors"
        )

    # Rigid-rotation clause check
    if "rigid object" not in lowered or "coherent surface" not in lowered:
        warnings.append(
            "missing rigid-rotation orientation clause "
            "('packaging is a rigid object … one coherent surface')"
        )

    # Two-leg + occlusion anatomy clause check
    if "two legs" not in lowered and "two arms" not in lowered:
        warnings.append(
            "missing anatomy sanity clause "
            "(should mention 'two arms, two hands, two legs, five fingers per hand')"
        )
    if "occluded" not in lowered and "hidden" not in lowered:
        warnings.append(
            "missing occlusion clause ('fingers and hands occluded by the product "
            "or her body still fully exist')"
        )

    # Single-product clause check
    if "exactly one" not in lowered and "one physical" not in lowered:
        warnings.append(
            "missing single-product clause "
            "('exactly ONE physical Alluvi product is visible')"
        )

    if warnings:
        print(f"[step_2_prompt_builder_ollama] WARNINGS for {scenario_id}:")
        for w in warnings:
            print(f"  ! {w}")


def build_step_2_prompt(scenario: dict, step_1_output: dict) -> dict:
    """
    Build the Qwen-tuned Step 2 prompt envelope via local Ollama.
    Same return shape as the Claude version.
    """
    if not QWEN_SYSTEM_PROMPT_PATH.exists():
        raise FileNotFoundError(f"missing system prompt: {QWEN_SYSTEM_PROMPT_PATH}")

    system_prompt = QWEN_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    ctx = _load_static_context()

    user_message = "\n".join(
        [
            "=== product.yaml (INTERNAL VALIDATION ONLY — never describe in prompt) ===",
            ctx["product_yaml"],
            "",
            "=== do_dont.md (compliance) ===",
            ctx["do_dont_md"],
            "",
            "=== ORIGINAL SCENARIO ===",
            json.dumps(scenario, indent=2),
            "",
            "=== STEP 1 OUTPUT (use product_slot for placement, lighting from sentence 4) ===",
            json.dumps(step_1_output, indent=2),
            "",
            "=== TASK ===",
            "Build the Qwen-tuned Step 2 prompt envelope.",
            "",
            "BREVITY IS REQUIRED. Past Qwen runs with 500+ word prompts performed WORSE",
            "  than 380-word versions of the same prompt. Stay between 320 and 410 words.",
            "  HARD CEILING is 430 words. Do NOT exceed it. If a clause is redundant with",
            "  another, drop it. If two clauses say the same thing, keep the shorter one.",
            "  Match the calibration examples' length (~387 words each). Do not be more",
            "  verbose than the examples — they were tuned for Qwen's signal-to-noise",
            "  characteristics.",
            "",
            "Use 'the person from the first image' / 'the product from the second image' /",
            "  'the first image' / 'the second image' positional reference syntax",
            "  (REQUIRED in this Qwen variant — do NOT use generic 'reference photo' language).",
            "Thread 'keep X unchanged' anchors through Sentence 1 (keep her face unchanged,",
            "  keep her hair unchanged, keep her outfit unchanged, keep the scene unchanged).",
            "Echo Step 1's lighting language from sentence 4 verbatim in Sentence 4.",
            "Sentence 2 MUST include the rigid-rotation orientation clause:",
            "  'The packaging is a rigid object — its proportions and printed design match",
            "   the second image exactly. The box can be rotated naturally for the holding",
            "   pose, but the printed design rotates with it as one coherent surface; never",
            "   reflow, redesign, or rearrange the layout to fit a different orientation;",
            "   never mirror or reverse the text.'",
            "  This REPLACES the older 'natural landscape orientation, do not rotate to",
            "  vertical' language. Rotation IS allowed; redesign/reflow IS NOT.",
            "Sentence 2 MUST include positive + negative position re-anchoring",
            "  (e.g. 'at chest level, not above her head, not at her hip').",
            "Sentence 3 MUST include the anatomy sanity clause with occlusion handling",
            "  (exactly two arms, two hands, TWO LEGS, five fingers per hand;",
            "   fingers and hands occluded by the product or her body still fully exist —",
            "   do not omit them because they are hidden; no extra limbs, no extra digits,",
            "   no fused or warped fingers).",
            "Sentence 4 MUST include the single product clause at its start",
            "  ('exactly ONE physical Alluvi product is visible — never two copies, never",
            "   duplicates'; for mirror-reflection scenarios add 'the mirror reflection",
            "   counts as the same product').",
            "Include the white base preservation clause in Sentence 4.",
            "",
            "Word count for step_2_image_prompt: 320-410 (HARD CEILING 430).",
            "",
            "CRITICAL OUTPUT RULES:",
            "  - Respond with a SINGLE JSON object only.",
            "  - No markdown fences (no ```json, no ```).",
            "  - No explanatory text before or after the JSON.",
            "  - Start your response with { and end with }.",
        ]
    )

    scenario_id = scenario.get("id", "?")
    print(
        f"[step_2_prompt_builder_ollama] Step 2 (Qwen-tuned) -> Ollama "
        f"({ollama_client.DEFAULT_MODEL}) for scenario {scenario_id}"
    )

    output = ollama_client.generate_json(
        prompt=user_message,
        system=system_prompt,
        options=OLLAMA_OPTIONS,
        required_keys=["step_2_image_prompt"],
    )

    # Word count: compute from actual prompt text (most reliable).
    prompt_text = output.get("step_2_image_prompt", "") or ""
    actual_wc = len(prompt_text.split())
    declared_wc = (
        output.get("word_count")
        or output.get("step_2_image_prompt_word_count")
    )
    wc_display = (
        f"{actual_wc}"
        if declared_wc is None or abs(declared_wc - actual_wc) <= 20
        else f"{actual_wc} (declared {declared_wc})"
    )
    print(f"[step_2_prompt_builder_ollama]   Step 2 done: word_count={wc_display}")

    _warn_if_weak(output, scenario_id)

    return output