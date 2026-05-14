"""
src/step_1_pulid.py — Stage 1 persona scene generation via local FLUX.1-dev + PuLID.

Replaces the previous fal-ai/flux-pulid caller. Delegates to
src/pulid_inference/pulid_pipeline.py which wraps guozinan/PuLID's
FluxGenerator (loaded from /workspace/PuLID).

Public surface (matches run.py expectations):
  load_pipeline()                              — heavy load (~28 GB VRAM)
  generate(pipeline, step_1_prompt, fal_pulid_params, out_path, scenario_id)

Return-dict contract matches the previous fal version (with fal_url=None,
request_id=local UUID, cost_usd=0.0) so DB writes / HTML viewers need no
changes.

PARAM MAPPING from the Opus `fal_pulid_params` envelope to local PuLID:
  image_size.{width,height}    → width, height
  num_inference_steps          → num_inference_steps
  guidance_scale               → guidance_scale
  id_weight (clamped to ≤1.0)  → id_weight
  true_cfg                     → true_cfg
  max_sequence_length          → max_sequence_length  (fallback 512, NOT 256 —
                                                       see step_1_prompt_builder.py)
  negative_prompt              → negative_prompt      (fallback DEFAULT_NEGATIVE_PROMPT)
  seed                         → seed (None becomes -1 = random inside PuLID)

Fal-only keys ignored locally: enable_safety_checker, output_format, num_images.

PER-CALL AUDIT:
  Before inference, writes `{out_path.stem}_request.json` next to out_path
  capturing the exact prompt + resolved params actually passed to the pipeline.
  Useful for post-hoc debugging — match what was sent against what came back.
"""

import json
import time
import uuid
from pathlib import Path

from src.pulid_inference.pulid_pipeline import PulidWrapper


REPO_ROOT = Path(__file__).resolve().parents[1]
PERSONA_IMAGE_PATH = REPO_ROOT / "assets" / "persona.jpg"

ENDPOINT_LABEL = "local/flux-pulid"
COST_PER_IMAGE_USD = 0.0

# Practical hard cap on id_weight. PuLID's Gradio slider allows 0-3.0, but
# above 1.0 the face starts to overfit and look pasted-on. Clamp here matches
# the existing fal-version behavior.
ID_WEIGHT_HARD_CAP = 1.0

# Defensive fallback: if the Opus envelope drifts and omits negative_prompt,
# use this hardcoded baseline tuned for natural skin + strict anatomy
# (2 arms / 2 legs / 5 fingers) + anti-cartoonish drift. Ported verbatim
# from the production fal-version step_1_pulid.py.
DEFAULT_NEGATIVE_PROMPT = (
    "plastic skin, airbrushed skin, waxy skin, porcelain skin, "
    "doll-like features, smooth artificial skin, extra limbs, "
    "extra arms, extra legs, three legs, three arms, extra hands, "
    "extra fingers, six fingers, seven fingers, fused fingers, "
    "missing fingers, deformed hands, mutated hands, distorted face, "
    "asymmetric eyes, dead eyes, uncanny valley, perfect symmetry, "
    "model pose, stiff pose, magazine retouching, AI-generated look, "
    "3D render, CGI, illustration, cartoon, anime, painting, lowres, "
    "blurry, jpeg artifacts, watermark, text, signature, logo"
)

# max_sequence_length default. MUST be 512 — at 256 the T5XXL encoder
# truncates at ~210 effective words, silently dropping sentence-5 camera
# anchor + identity-lock line (per step_1_prompt_builder.py docstring).
DEFAULT_MAX_SEQUENCE_LENGTH = 512


def load_pipeline() -> PulidWrapper:
    """Load FLUX.1-dev + PuLID + InsightFace. ~28 GB VRAM at bfloat16."""
    print("[step_1_pulid] loading PuLID pipeline (FLUX.1-dev + PuLID + InsightFace)")
    t0 = time.time()
    pipeline = PulidWrapper(model_name="flux-dev", version="v0.9.1")
    elapsed = time.time() - t0
    print(f"[step_1_pulid]   pipeline ready in {elapsed:.1f}s")
    return pipeline


def _parse_image_size(value, fallback=(768, 1344)) -> tuple[int, int]:
    """fal's image_size dict/preset → (width, height) tuple."""
    if isinstance(value, dict):
        w = value.get("width")
        h = value.get("height")
        if w and h:
            return int(w), int(h)
    presets = {
        "square_hd":      (1024, 1024),
        "square":         (512, 512),
        "portrait_4_3":   (768, 1024),
        "portrait_16_9":  (576, 1024),
        "landscape_4_3":  (1024, 768),
        "landscape_16_9": (1024, 576),
    }
    if isinstance(value, str) and value in presets:
        return presets[value]
    return fallback


def generate(
    pipeline: PulidWrapper,
    step_1_prompt: str,
    fal_pulid_params: dict,
    out_path: Path,
    scenario_id: str = "unknown",
) -> dict:
    """Run PuLID Stage 1 with a preloaded pipeline."""
    if not PERSONA_IMAGE_PATH.exists():
        raise FileNotFoundError(f"persona.jpg not found at {PERSONA_IMAGE_PATH}")

    # Resolve params from envelope
    width, height = _parse_image_size(fal_pulid_params.get("image_size"))
    num_steps = int(fal_pulid_params.get("num_inference_steps", 30))
    guidance = float(fal_pulid_params.get("guidance_scale", 3.5))
    true_cfg = float(fal_pulid_params.get("true_cfg", 1.5))

    max_seq_raw = fal_pulid_params.get("max_sequence_length", DEFAULT_MAX_SEQUENCE_LENGTH)
    try:
        max_seq_len = int(max_seq_raw)
    except (TypeError, ValueError):
        max_seq_len = DEFAULT_MAX_SEQUENCE_LENGTH

    # Defensive negative_prompt fallback (same pattern as fal-version)
    negative_prompt = fal_pulid_params.get("negative_prompt")
    if not negative_prompt:
        print(
            f"[step_1_pulid] [{scenario_id}] no negative_prompt in envelope, "
            f"using hardcoded default"
        )
        negative_prompt = DEFAULT_NEGATIVE_PROMPT

    seed = fal_pulid_params.get("seed")

    # Clamp id_weight at 1.0
    id_weight_raw = fal_pulid_params.get("id_weight", 1.0)
    try:
        id_weight = float(id_weight_raw)
        if id_weight > ID_WEIGHT_HARD_CAP:
            print(
                f"[step_1_pulid] WARNING: id_weight {id_weight} > cap "
                f"{ID_WEIGHT_HARD_CAP}, clamping"
            )
            id_weight = ID_WEIGHT_HARD_CAP
    except (TypeError, ValueError):
        print(
            f"[step_1_pulid] WARNING: id_weight {id_weight_raw!r} not numeric, "
            f"defaulting to 1.0"
        )
        id_weight = 1.0

    # ─── AUDIT: save exact request payload BEFORE inference ─────────────
    # Captures what was actually passed to the pipeline — match against output
    # for debugging. File lives next to the generated image, same stem +
    # "_request.json" suffix (e.g. 03_step1_persona_request.json).
    resolved_args = {
        "prompt": step_1_prompt,
        "persona_image_path": str(PERSONA_IMAGE_PATH),
        "width": width,
        "height": height,
        "num_inference_steps": num_steps,
        "guidance_scale": guidance,
        "id_weight": id_weight,
        "true_cfg": true_cfg,
        "negative_prompt": negative_prompt,
        "max_sequence_length": max_seq_len,
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
    print(f"[step_1_pulid] [{scenario_id}]   request audit -> {audit_path.name}")

    print(
        f"[step_1_pulid] [{scenario_id}] inferring "
        f"(size={width}x{height}, steps={num_steps}, guidance={guidance}, "
        f"id_weight={id_weight}, true_cfg={true_cfg}, "
        f"max_seq_len={max_seq_len}, seed={seed})"
    )

    t0 = time.time()
    image, used_seed = pipeline.generate(
        prompt=step_1_prompt,
        persona_image_path=str(PERSONA_IMAGE_PATH),
        width=width,
        height=height,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        id_weight=id_weight,
        true_cfg=true_cfg,
        negative_prompt=negative_prompt,
        max_sequence_length=max_seq_len,
        seed=seed,
    )
    elapsed = time.time() - t0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, "JPEG", quality=95)

    print(
        f"[step_1_pulid] [{scenario_id}]   rendered in {elapsed:.1f}s, "
        f"seed={used_seed}"
    )

    return {
        "local_path": str(out_path),
        "fal_url": None,
        "seed": used_seed,
        "request_id": f"local-{uuid.uuid4().hex[:8]}",
        "elapsed_seconds": elapsed,
        "endpoint": ENDPOINT_LABEL,
        "cost_usd": COST_PER_IMAGE_USD,
        "fal_pulid_params_used": {
            "image_size": {"width": width, "height": height},
            "num_inference_steps": num_steps,
            "guidance_scale": guidance,
            "true_cfg": true_cfg,
            "id_weight": id_weight,
            "max_sequence_length": max_seq_len,
            "negative_prompt": negative_prompt,
            "seed": seed,
        },
    }