"""
src/step_2_qwen_edit.py — local Qwen-Image-Edit-2511 product compositing.

Replaces the previous fal-ai/qwen-image-edit-2511 caller. Multi-image input
[persona_scene, product] is passed as a Python list to the
QwenImageEditPlusPipeline (the "Plus" variant supports multi-image).

Public surface:
  load_pipeline()           — heavy load (~40 GB VRAM). Call once per stage.
  generate(pipeline, ...)   — runs inference with a preloaded pipeline.

Return-dict contract matches the old fal version (with fal_url=None,
request_id=local UUID, cost_usd=0.0) so DB writes and HTML viewers
need no changes.

CFG requires BOTH true_cfg_scale > 1 AND negative_prompt:
  Per diffusers docs ("Classifier-free guidance is enabled by setting
  true_cfg_scale > 1 and a provided negative_prompt"), passing
  true_cfg_scale=4.0 without negative_prompt silently disables CFG.
  Every official Qwen-Image-Edit-2511 example passes negative_prompt=" "
  (single space). We do the same here, with a defensive fallback if the
  envelope omits it.

  Also: the `guidance_scale` kwarg is INEFFECTIVE on Qwen-Image-Edit-2511
  ("guidance_scale parameter is there to support future guidance-distilled
  models... Note that passing guidance_scale to the pipeline is ineffective").
  We don't pass it.

MAX_SEQUENCE_LENGTH FIX (CRITICAL for prompt fidelity):
  The pipeline's default max_sequence_length is 512 tokens (~200 words),
  with a hard ceiling of 1024. Prompts longer than 512 tokens are SILENTLY
  TRUNCATED — instructions at the end of the prompt are simply ignored by
  the model. For the Alluvi pipeline this matters: the Opus-generated Step 2
  prompts run 500-900 tokens, and the product preservation clauses are
  typically near the end (which gets cut off). Bumping to 1024 unlocks the
  full prompt and is the single biggest quality lever for product fidelity.

PER-CALL AUDIT:
  Before inference, writes `{out_path.stem}_request.json` next to out_path
  capturing the exact prompt + resolved params + image input paths.
"""

import json
import os
import time
import uuid
import torch
from pathlib import Path
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

# Project paths — repo root is two levels up from src/
REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCT_IMAGE_PATH = REPO_ROOT / "assets" / "product.jpg"

# Model location. Override via env var if needed.
DEFAULT_QWEN_PATH = os.environ.get(
    "QWEN_EDIT_MODEL_PATH",
    "/workspace/models/Qwen-Image-Edit-2511",
)

ENDPOINT_LABEL = "local/qwen-image-edit-2511"
COST_PER_IMAGE_USD = 0.0  # GPU time tracked separately at batch level

# Sensible defaults if scenario envelope doesn't specify.
# IMPORTANT: these match the official Qwen-Image-Edit-2511 README values.
DEFAULT_NUM_STEPS = 40           # official Qwen team recommendation
DEFAULT_TRUE_CFG_SCALE = 3.0     # official Qwen team recommendation

# CFG balance requirement — without this, true_cfg_scale silently does nothing.
# Single space is intentional (matches official Qwen examples). DO NOT change
# to empty string — empty string is treated as "no negative prompt" in
# diffusers and skips the negative branch entirely.
DEFAULT_NEGATIVE_PROMPT = " "

# Maximum allowed by the model is 1024 — bumping from the silently-truncating
# default 512 to capture the full Opus-generated prompt including product
# preservation clauses at the tail.
DEFAULT_MAX_SEQUENCE_LENGTH = 1024


def load_pipeline(model_path: str = DEFAULT_QWEN_PATH) -> QwenImageEditPlusPipeline:
    """
    Load Qwen-Image-Edit-2511 to GPU. ~40-48 GB VRAM at bfloat16.
    Takes ~30-60s from network volume on first call.
    """
    print(f"[step_2_qwen] loading pipeline from {model_path}")
    t0 = time.time()
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    elapsed = time.time() - t0
    print(f"[step_2_qwen]   pipeline ready in {elapsed:.1f}s")
    return pipe


def _parse_image_size(value) -> tuple[int, int] | None:
    """
    Translate fal's `image_size` (dict or preset string) to (width, height).
    Returns None if value can't be interpreted — caller falls back to input dims.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        w = value.get("width")
        h = value.get("height")
        if w and h:
            return int(w), int(h)
    presets = {
        "square_hd": (1024, 1024),
        "square": (512, 512),
        "portrait_4_3": (768, 1024),
        "portrait_16_9": (576, 1024),
        "landscape_4_3": (1024, 768),
        "landscape_16_9": (1024, 576),
    }
    if isinstance(value, str) and value in presets:
        return presets[value]
    return None


def generate(
    pipeline: QwenImageEditPlusPipeline,
    step_1_local_path: str,
    step_2_prompt: str,
    fal_qwen_params: dict,
    out_path: Path,
    scenario_id: str = "unknown",
) -> dict:
    """
    Run Qwen-Image-Edit-2511 on [persona scene, product] inputs.

    Args:
        pipeline: preloaded QwenImageEditPlusPipeline (from load_pipeline())
        step_1_local_path: path to the Step 1 PuLID output image
        step_2_prompt: Opus-generated Step 2 prompt text
        fal_qwen_params: dict from prompt envelope (image_size, num_inference_steps,
                         seed, etc.). Param dict keeps the fal_qwen_params name for
                         compatibility with the existing prompt builder output shape.
        out_path: where to write the resulting JPG
        scenario_id: for logging

    Returns:
        dict with local_path, fal_url, seed, request_id, elapsed_seconds,
        endpoint, cost_usd.
    """
    if not PRODUCT_IMAGE_PATH.exists():
        raise FileNotFoundError(f"product.jpg not found at {PRODUCT_IMAGE_PATH}")
    if not Path(step_1_local_path).exists():
        raise FileNotFoundError(f"Step 1 image not found at {step_1_local_path}")

    persona_image = Image.open(step_1_local_path).convert("RGB")
    product_image = Image.open(PRODUCT_IMAGE_PATH).convert("RGB")

    # Resolve params with sensible fallbacks. Qwen-Image-Edit uses
    # true_cfg_scale (not guidance_scale) but the envelope was named for fal,
    # so we accept both keys and prefer the more specific one.
    num_inference_steps = int(fal_qwen_params.get("num_inference_steps", DEFAULT_NUM_STEPS))
    true_cfg_scale = float(
        fal_qwen_params.get("true_cfg_scale",
                            fal_qwen_params.get("guidance_scale", DEFAULT_TRUE_CFG_SCALE))
    )
    negative_prompt = fal_qwen_params.get("negative_prompt")
    if not negative_prompt:
        negative_prompt = DEFAULT_NEGATIVE_PROMPT

    # ─── max_sequence_length unlock ─────────────────────────────────────
    # Default 512 silently truncates prompts >~200 words. Bumping to 1024
    # (the hard model ceiling) captures full Opus-generated prompts so the
    # product preservation clauses near the end of the prompt actually reach
    # the model. Single biggest quality lever for this pipeline.
    max_sequence_length = int(fal_qwen_params.get("max_sequence_length",
                                                    DEFAULT_MAX_SEQUENCE_LENGTH))
    # Clamp to model's hard ceiling
    if max_sequence_length > 1024:
        max_sequence_length = 1024

    seed = fal_qwen_params.get("seed")

    # image_size translation (fal dict/preset → diffusers width/height)
    size = _parse_image_size(fal_qwen_params.get("image_size"))
    width, height = size if size else (persona_image.width, persona_image.height)

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(int(seed))

    # ─── AUDIT: save exact request payload BEFORE inference ─────────────
    # image_urls order MUST match the Qwen-tuned prompt's references:
    #   "the person from the first image"   → image_urls[0] (Stage 1 output)
    #   "the product from the second image" → image_urls[1] (product.jpg)
    resolved_args = {
        "prompt": step_2_prompt,
        "image_urls": [str(Path(step_1_local_path).resolve()),
                       str(PRODUCT_IMAGE_PATH.resolve())],
        "num_inference_steps": num_inference_steps,
        "true_cfg_scale": true_cfg_scale,
        "negative_prompt": negative_prompt,
        "max_sequence_length": max_sequence_length,
        "width": width,
        "height": height,
        "seed": seed,
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
    print(f"[step_2_qwen] [{scenario_id}]   request audit -> {audit_path.name}")

    print(
        f"[step_2_qwen] [{scenario_id}] inferring "
        f"(steps={num_inference_steps}, true_cfg={true_cfg_scale}, "
        f"neg_prompt={negative_prompt!r}, max_seq={max_sequence_length}, "
        f"size={width}x{height}, seed={seed})"
    )

    t0 = time.time()
    result = pipeline(
        image=[persona_image, product_image],
        prompt=step_2_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=true_cfg_scale,
        max_sequence_length=max_sequence_length,
        height=height,
        width=width,
        generator=generator,
    )
    elapsed = time.time() - t0

    output_image = result.images[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_image.save(out_path, "JPEG", quality=95)

    print(f"[step_2_qwen] [{scenario_id}]   composited in {elapsed:.1f}s")

    return {
        "local_path": str(out_path),
        "fal_url": None,
        "seed": seed,
        "request_id": f"local-{uuid.uuid4().hex[:8]}",
        "elapsed_seconds": elapsed,
        "endpoint": ENDPOINT_LABEL,
        "cost_usd": COST_PER_IMAGE_USD,
    }