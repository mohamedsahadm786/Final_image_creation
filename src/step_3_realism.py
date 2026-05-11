"""
src/step_3_realism.py — Stage 3 photoreal refinement via FLUX.1 Kontext.

WHY KONTEXT (not img2img):
  img2img refinement drifts product text/typography at ANY strength > 0
  because text is high-frequency detail that gets nudged during each denoise
  step. FLUX.1 Kontext is purpose-built for surgical edits that PRESERVE
  typography, character identity, and unchanged regions. Black Forest Labs
  explicitly markets Kontext for "Typography and Text Editing — seamlessly
  edit text within images... preserves text styling when you change words."

WHAT KONTEXT DOES DIFFERENTLY:
  - It reads the image AND understands edit instructions
  - It applies the edit surgically while preserving everything else
  - No "strength" parameter — the model decides what to change
  - Prompts are INSTRUCTIONS ("make X natural"), not DESCRIPTIONS ("a photo of...")
  - Explicit "keep Y unchanged" clauses are the standard preservation mechanism

CRITICAL: SAFETY FILTER HANDLING
  Kontext's safety filter defaults to safety_tolerance="2" (very strict).
  Content showing skin (cleavage, midriff, bra, swimwear) triggers it and
  the filter SILENTLY returns an all-black image instead of refusing.
  We pass safety_tolerance="5" (max permissive — the documented maximum)
  to allow legitimate ad content showing typical robe/loungewear.

  We ALSO check the downloaded bytes — if the image is essentially all black
  (mean pixel < 5), we raise RuntimeError so the caller doesn't pass garbage
  to downstream stages. With Claude flow's QC retry loop, this triggers a
  clean retry rather than a silent failure.

WHEN THIS IS CALLED:
  - Claude flow (run.py):       AFTER QC passes — only good images get refined.
  - Ollama flow (run_ollama.py): unconditionally after Stage 2 succeeds.

WHAT THIS CAN FIX:
  - Waxy/plastic skin → natural skin with pores and small imperfections
  - Flat fabric → realistic weave and folds
  - "AI gloss" lighting → natural-looking lighting
  - Hair as a mass → individual strands visible

WHAT THIS CANNOT FIX:
  - Anatomy defects (caught by QC in Claude flow)
  - Unnatural poses or compositions (baked in earlier)
  - Major product text issues (Kontext is much better than img2img but
    still not pixel-perfect — for that we'd need masked inpainting)

Drop-in shape match for src/step_2_qwen_edit.py:
  - Same upload + cache pattern (shared cache/fal_uploads.json)
  - Same return-dict contract:
        local_path, fal_url, seed, request_id, elapsed_seconds, endpoint, cost_usd
"""

import os
import json
import time
import httpx
import fal_client
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

if not os.getenv("FAL_KEY"):
    raise RuntimeError("FAL_KEY not set in environment.")

# FLUX.1 Kontext Pro — instruction-based image editing.
# Specifically chosen over img2img because Kontext preserves typography,
# character identity, and unchanged regions by design.
#
# Alternatives if needed:
#   - fal-ai/flux-kontext/dev    (open-weights dev version, similar quality)
#   - fal-ai/flux-pro/kontext/max (more powerful, more expensive, STRICTER safety filter)
FAL_ENDPOINT = "fal-ai/flux-pro/kontext"

# Project paths — repo root is two levels up from this file
REPO_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_CACHE_PATH = REPO_ROOT / "cache" / "fal_uploads.json"

# Kontext Pro pricing: $0.04 per image (flat, not per-megapixel).
COST_PER_IMAGE_USD = 0.04

# Kontext defaults — these are the values from fal's official API docs.
DEFAULT_NUM_STEPS = 28
DEFAULT_GUIDANCE = 3.5

# Safety tolerance: "1" (strict) to "5" (permissive). Default "2" trips on
# ANY skin/cleavage/bra/midriff — even fully legitimate ad content with a
# robe and visible bra (which is what we have). "5" is the documented max
# and is required for our content category to pass.
DEFAULT_SAFETY_TOLERANCE = "5"

# Below this mean pixel value (0-255), the image is considered effectively
# all-black — meaning the safety filter triggered and silently zeroed it out.
BLACK_IMAGE_MEAN_THRESHOLD = 5.0

# Instruction-based prompt. Tells Kontext WHAT TO CHANGE and explicitly
# states WHAT TO PRESERVE — the preservation clauses are critical, the
# Kontext docs say "Explicitly state what should remain unchanged."
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


def _load_cache() -> dict:
    if UPLOAD_CACHE_PATH.exists():
        try:
            return json.loads(UPLOAD_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    UPLOAD_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    UPLOAD_CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _upload_with_cache(local_path: str) -> str:
    """Upload to fal once and cache the URL keyed by absolute path."""
    abs_path = str(Path(local_path).resolve())
    cache = _load_cache()
    if abs_path in cache and isinstance(cache[abs_path], str):
        return cache[abs_path]
    print(f"[step_3_realism] uploading: {abs_path}")
    url = fal_client.upload_file(abs_path)
    cache[abs_path] = url
    _save_cache(cache)
    return url


def _check_not_all_black(image_path: Path, scenario_id: str) -> None:
    """
    Defensive: if Kontext's safety filter triggered, the downloaded image
    will be all black. Detect this and raise so we don't silently pass
    garbage to downstream code.

    Raises RuntimeError if the image is effectively all-black.
    """
    try:
        with Image.open(image_path) as img:
            grayscale = img.convert("L")  # luminance channel
            # Sample mean of small subsample for speed on large images
            sample = grayscale.resize((64, 64))
            pixels = list(sample.getdata())
            mean_pixel = sum(pixels) / len(pixels)
    except Exception as e:
        raise RuntimeError(
            f"Could not inspect Kontext output for {scenario_id}: "
            f"{type(e).__name__}: {e}"
        )

    if mean_pixel < BLACK_IMAGE_MEAN_THRESHOLD:
        raise RuntimeError(
            f"Kontext returned an all-black image for {scenario_id} "
            f"(mean pixel = {mean_pixel:.2f}, threshold = {BLACK_IMAGE_MEAN_THRESHOLD}). "
            f"This means the safety filter triggered on the Stage 2 input "
            f"despite safety_tolerance=5. "
            f"Likely cause: Stage 2 image shows visible skin/cleavage/bra "
            f"that Kontext flags as NSFW even at max permissive setting. "
            f"Workaround: adjust Stage 2 prompts to keep the robe more closed, "
            f"OR disable Stage 3 for this scenario by setting STEP_3_ENABLED=false."
        )


def generate(
    step_2_local_path: str,
    out_path: Path,
    *,
    scenario_id: str = "unknown",
    extra_lighting_hint: str | None = None,
    num_inference_steps: int = DEFAULT_NUM_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE,
    safety_tolerance: str = DEFAULT_SAFETY_TOLERANCE,
    # `strength` is accepted but IGNORED for backward compatibility with
    # callers that still pass it. Kontext doesn't use strength.
    strength: float | None = None,
    # `image_size` accepted but IGNORED — Kontext infers from input.
    image_size: dict | None = None,
) -> dict:
    """
    Run FLUX.1 Kontext realism pass on the Stage 2 output.

    Args:
        step_2_local_path:   filesystem path to the Step 2 (Qwen) output image
        out_path:            where to write the resulting image
        scenario_id:         for logging
        extra_lighting_hint: optional scenario.lighting text — appended to the
                             instruction so Kontext keeps the intended scene
                             lighting tone (e.g. "soft warm morning daylight").
        num_inference_steps: Kontext default is 28 (good balance speed/quality)
        guidance_scale:      CFG scale — Kontext default is 3.5
        safety_tolerance:    "1" (strict) to "5" (permissive). Default "5".
                             We need "5" to pass content with visible robe/cleavage.
        strength:            ACCEPTED FOR BACKWARD COMPAT — IGNORED.
        image_size:          ACCEPTED FOR BACKWARD COMPAT — IGNORED.

    Returns:
        dict with local_path, fal_url, seed, request_id, elapsed_seconds,
        endpoint, cost_usd, prompt_used, safety_tolerance.

    Raises:
        FileNotFoundError:  if step_2_local_path doesn't exist
        RuntimeError:       if Kontext returns no URL OR returns an all-black
                            image (safety filter triggered despite tolerance=5)
    """
    if not Path(step_2_local_path).exists():
        raise FileNotFoundError(
            f"Step 2 image not found at {step_2_local_path} — "
            f"can't run realism pass on a missing input."
        )

    # Build the instruction. Append the scenario's lighting hint if provided —
    # tells Kontext to preserve the specific lighting mood.
    instruction = DEFAULT_REALISM_INSTRUCTION
    if extra_lighting_hint:
        instruction += f" The lighting should remain: {extra_lighting_hint.strip()}"

    print(f"[step_3_realism] [{scenario_id}] uploading Step 2 image to fal")
    step_2_url = _upload_with_cache(step_2_local_path)

    arguments: dict = {
        "image_url": step_2_url,
        "prompt": instruction,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "safety_tolerance": safety_tolerance,
        "output_format": "jpeg",
    }

    print(f"[step_3_realism] [{scenario_id}] calling {FAL_ENDPOINT}")
    print(
        f"[step_3_realism] [{scenario_id}]   steps={num_inference_steps}, "
        f"guidance={guidance_scale}, safety_tolerance={safety_tolerance}"
    )

    t0 = time.time()
    result = fal_client.subscribe(FAL_ENDPOINT, arguments=arguments, with_logs=False)
    elapsed = time.time() - t0

    # Kontext returns the result in slightly different shape depending on
    # endpoint version — handle both 'images' list and 'image' single.
    image_url = None
    if "images" in result and result["images"]:
        image_url = result["images"][0].get("url")
    elif "image" in result:
        image_data = result["image"]
        if isinstance(image_data, dict):
            image_url = image_data.get("url")
        elif isinstance(image_data, str):
            image_url = image_data

    if not image_url:
        raise RuntimeError(
            f"Kontext returned no image URL for {scenario_id}. "
            f"Result keys: {list(result.keys())}, result: {str(result)[:500]}"
        )

    seed = result.get("seed")
    request_id = result.get("request_id", "unknown")

    print(
        f"[step_3_realism] [{scenario_id}]   refined in {elapsed:.1f}s, seed={seed}"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    response = httpx.get(image_url, timeout=180)
    response.raise_for_status()
    out_path.write_bytes(response.content)

    # Defensive: detect the silent-black-image safety filter case
    _check_not_all_black(out_path, scenario_id)

    return {
        "local_path": str(out_path),
        "fal_url": image_url,
        "seed": seed,
        "request_id": request_id,
        "elapsed_seconds": elapsed,
        "endpoint": FAL_ENDPOINT,
        "cost_usd": COST_PER_IMAGE_USD,
        "prompt_used": instruction,
        "safety_tolerance": safety_tolerance,
        "model_paradigm": "kontext_instruction_based",
    }