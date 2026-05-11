"""
src/step_1_pulid.py — Stage 1 persona scene generation via fal-ai/flux-pulid.

Drop-in shape matches src/step_2_qwen_edit.py:
  - Uploads reference image (persona.jpg) with cache
  - Calls fal endpoint with prompt + reference_image_url + tunable params
  - Returns the same dict contract: local_path, fal_url, seed, request_id,
    elapsed_seconds, endpoint, cost_usd

Per-scenario params come from the Opus Step 1 prompt envelope's
`fal_pulid_params` field — those override config.yaml defaults. The
critical per-scenario knobs are:
  - id_weight (1.0 for persona shots, 0.5 for flat-lays)  HARD CAP AT 1.0
  - true_cfg (1.5 medium, 1.7 close-up, 1.2 wide)
  - guidance_scale (3.5 default; lower for softer skin)
  - negative_prompt (locked baseline against AI-cartoonish drift)

The id_weight clamp is mandatory — fal API returns HTTP 422 if id_weight > 1.0.
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

FAL_ENDPOINT = "fal-ai/flux-pulid"

# Project paths
REPO_ROOT = Path(__file__).resolve().parents[1]
PERSONA_IMAGE_PATH = REPO_ROOT / "assets" / "persona.jpg"
UPLOAD_CACHE_PATH = REPO_ROOT / "cache" / "fal_uploads.json"

COST_PER_IMAGE_USD = 0.04  # PuLID — production-verified rate

# fal API hard cap on id_weight — values above 1.0 return HTTP 422
ID_WEIGHT_HARD_CAP = 1.0


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
    print(f"[step_1_pulid] uploading: {abs_path}")
    url = fal_client.upload_file(abs_path)
    cache[abs_path] = url
    _save_cache(cache)
    return url


def generate(
    step_1_prompt: str,
    fal_pulid_params: dict,
    out_path: Path,
    scenario_id: str = "unknown",
) -> dict:
    """
    Run PuLID with the persona reference + the Step 1 prompt.

    Args:
        step_1_prompt: the step_1_image_prompt string from the Opus envelope
        fal_pulid_params: dict of PuLID params from the Opus envelope
                          (image_size, num_inference_steps, guidance_scale,
                           id_weight, true_cfg, negative_prompt, etc.)
        out_path: where to write the resulting JPG
        scenario_id: for logging

    Returns:
        dict with local_path, fal_url, seed, request_id, elapsed_seconds,
        endpoint, cost_usd, plus the resolved fal_pulid_params used.
    """
    if not PERSONA_IMAGE_PATH.exists():
        raise FileNotFoundError(f"persona.jpg not found at {PERSONA_IMAGE_PATH}")

    persona_url = _upload_with_cache(str(PERSONA_IMAGE_PATH))

    # Build arguments dict, starting with the prompt + reference, then layering
    # in the Opus-emitted params with clamping for id_weight.
    arguments: dict = {
        "prompt": step_1_prompt,
        "reference_image_url": persona_url,
    }

    # Pass through every supported optional param from the envelope.
    # Clamp id_weight at 1.0 to avoid HTTP 422.
    passthrough_keys = (
        "image_size",
        "num_inference_steps",
        "guidance_scale",
        "true_cfg",
        "max_sequence_length",
        "num_images",
        "output_format",
        "enable_safety_checker",
        "negative_prompt",
        "seed",
    )
    for key in passthrough_keys:
        if key in fal_pulid_params:
            arguments[key] = fal_pulid_params[key]

    if "id_weight" in fal_pulid_params:
        requested = fal_pulid_params["id_weight"]
        try:
            requested_f = float(requested)
            if requested_f > ID_WEIGHT_HARD_CAP:
                print(
                    f"[step_1_pulid] WARNING: id_weight {requested_f} exceeds "
                    f"fal API cap {ID_WEIGHT_HARD_CAP}, clamping"
                )
                arguments["id_weight"] = ID_WEIGHT_HARD_CAP
            else:
                arguments["id_weight"] = requested_f
        except (TypeError, ValueError):
            print(
                f"[step_1_pulid] WARNING: id_weight {requested!r} not numeric, "
                f"falling back to default 1.0"
            )
            arguments["id_weight"] = 1.0

    img_size = arguments.get("image_size", "default")
    id_w = arguments.get("id_weight", "default")
    tcfg = arguments.get("true_cfg", "default")
    print(f"[step_1_pulid] [{scenario_id}] calling {FAL_ENDPOINT}")
    print(
        f"[step_1_pulid] [{scenario_id}]   image_size={img_size}, "
        f"id_weight={id_w}, true_cfg={tcfg}"
    )

    t0 = time.time()
    result = fal_client.subscribe(FAL_ENDPOINT, arguments=arguments, with_logs=False)
    elapsed = time.time() - t0

    images = result.get("images") or []
    if not images:
        raise RuntimeError(
            f"PuLID returned no images for {scenario_id}. "
            f"Result keys: {list(result.keys())}"
        )

    image_url = images[0].get("url")
    if not image_url:
        raise RuntimeError(
            f"PuLID image entry missing url for {scenario_id}: {images[0]}"
        )

    seed = result.get("seed")
    request_id = result.get("request_id", "unknown")

    print(
        f"[step_1_pulid] [{scenario_id}]   rendered in {elapsed:.1f}s, seed={seed}"
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
        "fal_pulid_params_used": arguments,
    }