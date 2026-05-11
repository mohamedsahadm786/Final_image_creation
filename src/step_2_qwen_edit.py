"""
src/step_2_qwen_edit.py — Stage 2 product compositing via fal-ai/qwen-image-edit-2511.

Drop-in shape match for src/step_1_pulid.py:
  - Uses the same product upload + cache pattern (shared cache/fal_uploads.json,
    keyed by absolute path)
  - Same image_urls order: [step_1_scene_url, product_url]
  - Same return-dict contract:
        local_path, fal_url, seed, request_id, elapsed_seconds, endpoint, cost_usd

Differences from PuLID caller:
  - Endpoint is fal-ai/qwen-image-edit-2511
  - Takes two reference images (persona scene + product) via image_urls
  - Parameters: prompt + image_urls + image_size + num_images + enable_safety_checker
  - NO negative_prompt is passed even if present in params — Qwen-Image-Edit-2511
    does NOT use classifier-free guidance, so negative_prompt has zero effect
"""

import os
import json
import time
import httpx
import fal_client
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("FAL_KEY"):
    raise RuntimeError("FAL_KEY not set in environment.")

FAL_ENDPOINT = "fal-ai/qwen-image-edit-2511"

# Project paths — repo root is two levels up from this file
REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCT_IMAGE_PATH = REPO_ROOT / "assets" / "product.jpg"
UPLOAD_CACHE_PATH = REPO_ROOT / "cache" / "fal_uploads.json"

COST_PER_IMAGE_USD = 0.04  # approximate


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
    print(f"[step_2_qwen] uploading: {abs_path}")
    url = fal_client.upload_file(abs_path)
    cache[abs_path] = url
    _save_cache(cache)
    return url


def generate(
    step_1_local_path: str,
    step_2_prompt: str,
    fal_qwen_params: dict,
    out_path: Path,
    scenario_id: str = "unknown",
) -> dict:
    """
    Run Qwen-Image-Edit-2511 on the persona scene + product reference.

    Args:
        step_1_local_path: filesystem path to the Step 1 PuLID output image
        step_2_prompt: the Opus-generated Step 2 prompt text
        fal_qwen_params: dict of Qwen-specific parameters (image_size,
                         num_images, output_format, enable_safety_checker, etc.)
        out_path: where to write the resulting image
        scenario_id: for logging

    Returns:
        dict with local_path, fal_url, seed, request_id, elapsed_seconds,
        endpoint, cost_usd.
    """
    if not PRODUCT_IMAGE_PATH.exists():
        raise FileNotFoundError(f"product.jpg not found at {PRODUCT_IMAGE_PATH}")

    product_url = _upload_with_cache(str(PRODUCT_IMAGE_PATH))

    print(f"[step_2_qwen] [{scenario_id}] uploading Step 1 scene to fal")
    step_1_url = _upload_with_cache(step_1_local_path)

    # image_urls order: scene first, product second.
    # The Qwen-tuned prompt refers to:
    #   "the person from the first image"   → image_urls[0]
    #   "the product from the second image" → image_urls[1]
    image_urls = [step_1_url, product_url]

    arguments: dict = {
        "prompt": step_2_prompt,
        "image_urls": image_urls,
    }

    # Pass through Qwen-supported optional params if present in config/envelope.
    # NOTE: negative_prompt is passed through to match production behavior, but
    # Qwen-Image-Edit-2511 does NOT use CFG — even when present, the model
    # ignores it (confirmed via Alibaba's PromptMaster blog). The fal API
    # accepts the param either way.
    passthrough_keys = (
        "image_size",
        "num_images",
        "output_format",
        "enable_safety_checker",
        "num_inference_steps",
        "guidance_scale",
        "acceleration",
        "negative_prompt",
        "seed",
    )
    for key in passthrough_keys:
        if key in fal_qwen_params:
            arguments[key] = fal_qwen_params[key]

    img_size = arguments.get("image_size", "default")
    print(f"[step_2_qwen] [{scenario_id}] calling {FAL_ENDPOINT}")
    print(f"[step_2_qwen] [{scenario_id}]   image_size={img_size}")

    t0 = time.time()
    result = fal_client.subscribe(FAL_ENDPOINT, arguments=arguments, with_logs=False)
    elapsed = time.time() - t0

    images = result.get("images") or []
    if not images:
        raise RuntimeError(
            f"Qwen-Image-Edit-2511 returned no images for {scenario_id}. "
            f"Result keys: {list(result.keys())}"
        )

    image_url = images[0].get("url")
    if not image_url:
        raise RuntimeError(
            f"Qwen image entry missing url for {scenario_id}: {images[0]}"
        )

    seed = result.get("seed")
    request_id = result.get("request_id", "unknown")

    print(
        f"[step_2_qwen] [{scenario_id}]   composited in {elapsed:.1f}s, seed={seed}"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    response = httpx.get(image_url, timeout=180)
    response.raise_for_status()
    out_path.write_bytes(response.content)

    return {
        "local_path": str(out_path),
        "fal_url": image_url,
        "seed": seed,
        "request_id": request_id,
        "elapsed_seconds": elapsed,
        "endpoint": FAL_ENDPOINT,
        "cost_usd": COST_PER_IMAGE_USD,
    }