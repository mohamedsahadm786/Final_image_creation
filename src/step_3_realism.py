"""
src/step_3_realism.py — local FLUX.1-Kontext-dev photoreal refinement.

Replaces the previous fal-ai/flux-pro/kontext caller. Kontext-dev has no
built-in safety filter, so we drop:
  - safety_tolerance param
  - _check_not_all_black() detector
  - fal upload + cache

The bedroom_robe black-image issue is gone (no filter = no black images).

WHY KONTEXT (not img2img):
  Kontext is instruction-based editing that preserves typography, character
  identity, and unchanged regions. img2img at any strength drifts product
  text ("ALLUVI" → "ALUUVI"). Kontext fixes that by design.

ASPECT RATIO PRESERVATION:
  Diffusers FluxKontextPipeline defaults to 1024×1024 and silently crops
  the input if width/height aren't passed explicitly (diffusers issue
  #11886). We read the input image's dimensions and pass them through —
  output matches input aspect ratio verbatim. For 9:16 TikTok-format
  scenarios (768×1344) both dims are already multiples of 16, so no
  rounding occurs.

Public surface:
  load_pipeline()           — heavy load (~24 GB VRAM). Call once per stage.
  generate(pipeline, ...)   — runs inference with a preloaded pipeline.

Return-dict contract matches the old fal version so DB / HTML viewers
need no changes (fal_url=None, request_id=local UUID, cost_usd=0.0).

PER-CALL AUDIT:
  Before inference, writes `{out_path.stem}_request.json` next to out_path
  capturing the full instruction (with lighting hint appended if any) + params.
"""

import json
import os
import time
import uuid
import torch
from pathlib import Path
from PIL import Image
from diffusers import FluxKontextPipeline

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_KONTEXT_PATH = os.environ.get(
    "KONTEXT_MODEL_PATH",
    "/workspace/models/FLUX.1-Kontext-dev",
)

ENDPOINT_LABEL = "local/flux-kontext-dev"
COST_PER_IMAGE_USD = 0.0  # GPU time tracked separately at batch level

# Kontext defaults (matching the previous fal-side defaults)
DEFAULT_NUM_STEPS = 28
DEFAULT_GUIDANCE = 3.5

# Instruction-based prompt. Tells Kontext WHAT TO CHANGE and explicitly states
# WHAT TO PRESERVE — the preservation clauses are critical (Kontext docs:
# "Explicitly state what should remain unchanged").
DEFAULT_REALISM_INSTRUCTION = (
    "Aggressively transform this image to look like an authentic candid "
    "smartphone photograph taken by a real person. "
    "Skin: hyper-realistic with prominent visible pores, fine vellus facial hair, "
    "natural under-eye softness, subsurface scattering, slight redness in cheeks "
    "and ears, micro-imperfections, NOT smooth and NOT waxy. "
    "Hair: individual strands clearly visible with natural flyaway pieces, "
    "realistic shine and shadow, NOT a smooth mass. "
    "Fabric: visible weave, realistic folds, natural texture variations. "
    "Lighting: real-world directional light with natural falloff and ambient "
    "occlusion in corners. "
    "Film characteristics: subtle grain, slight chromatic aberration at edges, "
    "natural color depth. "
    "ABSOLUTELY PRESERVE UNCHANGED: the exact composition, exact pose, "
    "exact facial identity and features, exact product packaging and all its "
    "text/labels/colors/layout, exact outfit, exact background. "
    "Remove all AI artifacts: no plastic skin, no glossy CGI look, no airbrushed "
    "appearance, no uncanny valley."
)


def load_pipeline(model_path: str = DEFAULT_KONTEXT_PATH) -> FluxKontextPipeline:
    """
    Load FLUX.1-Kontext-dev to GPU. ~24 GB VRAM at bfloat16.
    """
    print(f"[step_3_realism] loading pipeline from {model_path}")
    t0 = time.time()
    pipe = FluxKontextPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    elapsed = time.time() - t0
    print(f"[step_3_realism]   pipeline ready in {elapsed:.1f}s")
    return pipe


def generate(
    pipeline: FluxKontextPipeline,
    step_2_local_path: str,
    out_path: Path,
    *,
    scenario_id: str = "unknown",
    extra_lighting_hint: str | None = None,
    num_inference_steps: int = DEFAULT_NUM_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE,
    seed: int | None = None,
    # Accepted for backward-compat with the old fal-version signature; ignored.
    safety_tolerance: str | None = None,
    strength: float | None = None,
    image_size: dict | None = None,
) -> dict:
    """
    Run FLUX.1-Kontext-dev realism pass on the Stage 2 output.

    Args:
        pipeline: preloaded FluxKontextPipeline (from load_pipeline())
        step_2_local_path: path to the Step 2 (Qwen) output image
        out_path: where to write the resulting JPG
        scenario_id: for logging
        extra_lighting_hint: optional scenario.lighting text appended to the
                             instruction so Kontext preserves intended mood
        num_inference_steps: 28 default
        guidance_scale: 3.5 default
        seed: optional for reproducibility
        safety_tolerance / strength / image_size: ignored, kept for API compat

    Returns:
        dict with local_path, fal_url, seed, request_id, elapsed_seconds,
        endpoint, cost_usd, prompt_used, model_paradigm.
    """
    if not Path(step_2_local_path).exists():
        raise FileNotFoundError(
            f"Step 2 image not found at {step_2_local_path} — "
            f"can't run realism pass on a missing input."
        )

    # Build the instruction with optional lighting echo
    instruction = DEFAULT_REALISM_INSTRUCTION
    if extra_lighting_hint:
        instruction += f" The lighting should remain: {extra_lighting_hint.strip()}"

    input_image = Image.open(step_2_local_path).convert("RGB")
    # ─── ASPECT-RATIO FIX ───────────────────────────────────────────────
    # Diffusers FluxKontextPipeline defaults to 1024×1024 and crops the
    # input if width/height not passed. Read the input's dims and pass them
    # through so Stage 3 output matches Stage 2 aspect ratio exactly.
    input_w, input_h = input_image.size

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(int(seed))

    # ─── AUDIT: save exact request payload BEFORE inference ─────────────
    resolved_args = {
        "prompt": instruction,
        "image_input": str(Path(step_2_local_path).resolve()),
        "width": input_w,
        "height": input_h,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "extra_lighting_hint": extra_lighting_hint,
    }
    audit_path = out_path.parent / f"{out_path.stem}_request.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(
            {"endpoint": ENDPOINT_LABEL, "arguments": resolved_args},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[step_3_realism] [{scenario_id}]   request audit -> {audit_path.name}")

    print(
        f"[step_3_realism] [{scenario_id}] inferring "
        f"(size={input_w}x{input_h}, steps={num_inference_steps}, "
        f"guidance={guidance_scale}, seed={seed})"
    )

    t0 = time.time()
    result = pipeline(
        image=input_image,
        prompt=instruction,
        width=input_w,
        height=input_h,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    elapsed = time.time() - t0

    output_image = result.images[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_image.save(out_path, "JPEG", quality=95)

    print(f"[step_3_realism] [{scenario_id}]   refined in {elapsed:.1f}s")

    return {
        "local_path": str(out_path),
        "fal_url": None,
        "seed": seed,
        "request_id": f"local-{uuid.uuid4().hex[:8]}",
        "elapsed_seconds": elapsed,
        "endpoint": ENDPOINT_LABEL,
        "cost_usd": COST_PER_IMAGE_USD,
        "prompt_used": instruction,
        "safety_tolerance": None,  # not applicable to Kontext-dev
        "model_paradigm": "kontext_instruction_based",
    }