"""
ollama_flow/src/step_1_prompt_builder_ollama.py — Step 1 prompt via Ollama.

Mirrors the public API of the production Claude builder:
  build_step_1_prompt(scenario) -> dict

Reads master prompts and static context from the PARENT repo (one level up):
  ../../prompts/master_prompt_step1.md
  ../../assets/persona.yaml
  ../../brand/brand.yaml
  ../../brand/do_dont.md

This means the Ollama flow runs against the same prompts and same assets as
the production Claude flow. Only the LLM call differs.

Cost: $0 per call (runs locally via Ollama).

Quality expectations: a 7B local model will produce noticeably weaker prompts
than Opus 4.7. Common failure modes:
  - Word count drift (Step 1 budget is 130-160, expect 80-220)
  - Persona descriptor paraphrasing instead of verbatim copying
  - Missing photoreal anchors
  - Occasional JSON malformation (caught by ollama_client._extract_json)
  - Hallucinating product mention despite "NO product" rule

This builder logs WARNINGS when any of these are detected, so you can see
which scenarios got weak prompts without inspecting every output.
"""

import json
from pathlib import Path

from . import ollama_client


# Path math:
#   this file lives at  Final_Image_generation/ollama_flow/src/step_1_prompt_builder_ollama.py
#   parent repo root is Final_Image_generation/    (two levels up from this file)
PARENT_REPO_ROOT = Path(__file__).resolve().parents[2]

STEP_1_SYSTEM_PATH = PARENT_REPO_ROOT / "prompts" / "master_prompt_step1.md"
PERSONA_YAML_PATH = PARENT_REPO_ROOT / "assets" / "persona.yaml"
BRAND_YAML_PATH = PARENT_REPO_ROOT / "brand" / "brand.yaml"
DO_DONT_MD_PATH = PARENT_REPO_ROOT / "brand" / "do_dont.md"


_static_context_cache: dict[str, str] | None = None


# Ollama generation options tuned for structured JSON output
OLLAMA_OPTIONS = {
    "temperature": 0.3,          # low — small models drift wildly at 0.7+
    "top_p": 0.9,
    "num_predict": 4096,         # generous output budget
    "repeat_penalty": 1.1,
}


def _load_static_context() -> dict[str, str]:
    """Load and cache the three static context files Step 1 needs."""
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


def _warn_if_weak(output: dict, scenario_id: str) -> None:
    """Emit visible warnings when Ollama output looks weak.

    Note: validity of `step_1_image_prompt` is already enforced by the
    JSONSanityError raised in ollama_client.generate_json. This function
    catches softer issues — word count drift, schema variants, banned terms,
    missing photoreal anchors.
    """
    warnings = []

    # fal_pulid_params is recommended but not required — pipeline falls back
    # to config.yaml defaults if missing. Note as info, not warning.
    info = []
    if not output.get("fal_pulid_params"):
        info.append(
            "fal_pulid_params not emitted — using config.yaml defaults "
            "(scenario-specific tuning lost)"
        )

    # Word count check — compute from actual prompt text.
    # Accept either declared schema variant.
    prompt_text = output.get("step_1_image_prompt", "") or ""
    word_count = len(prompt_text.split())
    declared = (
        output.get("word_count")
        or output.get("step_1_image_prompt_word_count")
    )

    if word_count < 100:
        warnings.append(
            f"step_1_image_prompt is suspiciously short ({word_count} words, "
            f"expected 130-160 normal / 200-250 close-up)"
        )
    elif word_count > 280:
        warnings.append(
            f"step_1_image_prompt is over-long ({word_count} words, "
            f"expected 130-160 normal / 200-250 close-up)"
        )
    if declared is not None and abs(declared - word_count) > 20:
        warnings.append(
            f"declared word_count ({declared}) disagrees with actual ({word_count})"
        )

    # Banned content check
    banned = ("alluvi", "tirzepatide", "the product", "the box", "the package")
    lowered = prompt_text.lower()
    for term in banned:
        if term in lowered:
            warnings.append(
                f"prompt contains banned term '{term}' "
                f"(Step 1 must not mention the product)"
            )

    # Photoreal anchor check
    if "candid amateur smartphone snapshot" not in lowered:
        warnings.append(
            "missing early photoreal anchor 'candid amateur smartphone snapshot'"
        )
    if "real photograph, not ai-generated" not in lowered:
        warnings.append(
            "missing late photoreal anchor 'Real photograph, not AI-generated'"
        )

    # id_weight sanity
    pulid_params = output.get("fal_pulid_params") or {}
    id_weight = pulid_params.get("id_weight")
    if id_weight is not None:
        try:
            if float(id_weight) > 1.0:
                warnings.append(
                    f"id_weight {id_weight} exceeds fal API cap 1.0 "
                    f"(will be clamped by step_1_pulid)"
                )
        except (TypeError, ValueError):
            warnings.append(f"id_weight {id_weight!r} is not numeric")

    if info:
        print(f"[step_1_prompt_builder_ollama] INFO for {scenario_id}:")
        for i in info:
            print(f"  ~ {i}")
    if warnings:
        print(
            f"[step_1_prompt_builder_ollama] WARNINGS for {scenario_id}:"
        )
        for w in warnings:
            print(f"  ! {w}")


def build_step_1_prompt(scenario: dict) -> dict:
    """
    Build the Step 1 prompt envelope via local Ollama.
    Same return shape as the Claude version.
    """
    if not STEP_1_SYSTEM_PATH.exists():
        raise FileNotFoundError(f"missing system prompt: {STEP_1_SYSTEM_PATH}")

    system_prompt = STEP_1_SYSTEM_PATH.read_text(encoding="utf-8")
    ctx = _load_static_context()

    # Same user_message structure as the Claude version. We ADD an explicit
    # "Output ONLY a JSON object" reminder at the end because small models
    # are far more likely to add explanatory prose around the JSON.
    user_message = "\n".join(
        [
            "=== persona.yaml (USE prompt_descriptors VERBATIM) ===",
            ctx["persona_yaml"],
            "",
            "=== brand.yaml (vibe + palette context) ===",
            ctx["brand_yaml"],
            "",
            "=== do_dont.md (compliance rules) ===",
            ctx["do_dont_md"],
            "",
            "=== SCENARIO ===",
            json.dumps(scenario, indent=2),
            "",
            "=== TASK ===",
            "Build the Step 1 prompt envelope for this scenario.",
            "Output JSON matching the v2 Step 1 schema.",
            "Use scenario.outfit verbatim — never the persona reference photo's outfit.",
            "Use persona.yaml prompt_descriptors verbatim — never paraphrase.",
            "NO product mentioned. NO placeholder. The product hand is empty.",
            "Word count for step_1_image_prompt: 130-160 (200-250 acceptable for close-ups).",
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
        f"[step_1_prompt_builder_ollama] Step 1 -> Ollama "
        f"({ollama_client.DEFAULT_MODEL}) for scenario {scenario_id}"
    )

    output = ollama_client.generate_json(
        prompt=user_message,
        system=system_prompt,
        options=OLLAMA_OPTIONS,
        required_keys=["step_1_image_prompt"],
    )

    # Word count: compute from actual prompt text (most reliable). Also accept
    # either declared schema variant — qwen2.5:7b sometimes emits
    # `step_1_image_prompt_word_count` instead of the spec's `word_count`.
    prompt_text = output.get("step_1_image_prompt", "") or ""
    actual_wc = len(prompt_text.split())
    declared_wc = (
        output.get("word_count")
        or output.get("step_1_image_prompt_word_count")
    )
    wc_display = (
        f"{actual_wc}"
        if declared_wc is None or abs(declared_wc - actual_wc) <= 20
        else f"{actual_wc} (declared {declared_wc})"
    )
    slot_type = (output.get("product_slot") or {}).get("type", "?")
    print(
        f"[step_1_prompt_builder_ollama]   Step 1 done: "
        f"word_count={wc_display}, slot_type={slot_type}"
    )

    _warn_if_weak(output, scenario_id)

    return output