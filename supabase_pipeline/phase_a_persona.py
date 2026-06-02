"""
supabase_pipeline/phase_a_persona.py — Phase A reference-portrait generation via
plain FLUX.1-dev text-to-image (diffusers FluxPipeline).

This INVENTS a face from text (no input photo) — that's why it is NOT PuLID.
PuLID always conditions on an existing face; Phase A creates the face that PuLID
will later lock across every scenario.

Public surface mirrors the other stage modules so the orchestrator stays uniform:
  load_pipeline()                          — heavy load (~24 GB VRAM)
  generate(pipeline, portrait_prompt, out_path, ...) -> dict
  unload_pipeline(pipeline)                — free VRAM

Return-dict contract (same shape as step_1_pulid / step_2 / step_3):
  {local_path, fal_url(None), seed, request_id("local-<uuid8>"),
   elapsed_seconds, endpoint, cost_usd(0.0)}

FLUX-SPECIFIC NOTES (verified against current FLUX.1-dev guidance):
  - FLUX.1-dev is GUIDANCE-DISTILLED. The plain FluxPipeline does NOT take a
    negative prompt (passing one errors / is ignored). Realism is steered by
    POSITIVE phrasing already baked into the portrait_prompt + guidance_scale.
  - guidance_scale ~3.5, steps ~28-40 is the balanced range. Higher guidance =
    better prompt adherence but waxier skin; lower = more natural texture.
  - T5-XXL handles up to 512 tokens (max_sequence_length=512). The CLIP 77-token
    warning is harmless — the long prompt still lands via T5.
  - Loads the diffusers-format folder at /workspace/models/FLUX.1-dev directly.

PER-CALL AUDIT:
  Writes `{out_path.stem}_request.json` next to out_path with the exact prompt +
  resolved params, same convention as the other stages.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import torch
from diffusers import FluxPipeline

FLUX_DEV_DIR = os.environ.get("FLUX_DEV_MODEL_PATH", "/workspace/models/FLUX.1-dev")

ENDPOINT_LABEL = "local/flux-dev-txt2img"
COST_PER_IMAGE_USD = 0.0

# Defaults (research-backed; overridable per call).
DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 1024              # 3:4 portrait — head-and-shoulders reference
DEFAULT_NUM_STEPS = 30             # 28-40 balanced range
DEFAULT_GUIDANCE = 3.5             # 3.0-3.8 balanced range
DEFAULT_MAX_SEQUENCE_LENGTH = 512  # T5-XXL full context for FLUX.1-dev


def load_pipeline() -> FluxPipeline:
    """Load FLUX.1-dev for text-to-image. ~24 GB VRAM at bf16, fully on GPU."""
    print(f"[phase_a_persona] loading FLUX.1-dev (txt2img) from {FLUX_DEV_DIR}")
    t0 = time.time()
    pipe = FluxPipeline.from_pretrained(FLUX_DEV_DIR, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    print(f"[phase_a_persona]   pipeline ready in {time.time() - t0:.1f}s")
    return pipe


def unload_pipeline(pipeline: FluxPipeline | None) -> None:
    """Free VRAM (the orchestrator can also use vram_utils.unload_pipeline)."""
    if pipeline is None:
        return
    try:
        pipeline.to("cpu")
    except Exception as e:
        print(f"[phase_a_persona] .to('cpu') partial move: {e}")
    del pipeline
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate(
    pipeline: FluxPipeline,
    portrait_prompt: str,
    out_path: Path,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    num_inference_steps: int = DEFAULT_NUM_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    seed: int | None = None,
    account_id: str = "unknown",
) -> dict:
    """Render one reference portrait with a preloaded FLUX.1-dev pipeline.

    No negative prompt is passed — FLUX.1-dev is guidance-distilled and the
    plain pipeline does not use one. Realism lives in the positive prompt.
    """
    out_path = Path(out_path)

    # Resolve seed (record the one actually used so the portrait is reproducible).
    if seed is None:
        seed = uuid.uuid4().int % (2 ** 31)
    seed = int(seed)
    generator = torch.Generator("cuda").manual_seed(seed)

    # ─── AUDIT: save exact request payload BEFORE inference ───────────────
    resolved_args = {
        "prompt": portrait_prompt,
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "max_sequence_length": max_sequence_length,
        "seed": seed,
        "negative_prompt": None,  # guidance-distilled — not used
        "model_path": FLUX_DEV_DIR,
    }
    audit_path = out_path.parent / f"{out_path.stem}_request.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps({"endpoint": ENDPOINT_LABEL, "arguments": resolved_args},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[phase_a_persona] [{account_id}]   request audit -> {audit_path.name}")
    print(f"[phase_a_persona] [{account_id}] inferring "
          f"(size={width}x{height}, steps={num_inference_steps}, "
          f"guidance={guidance_scale}, seed={seed})")

    t0 = time.time()
    image = pipeline(
        prompt=portrait_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        max_sequence_length=max_sequence_length,
        generator=generator,
    ).images[0]
    elapsed = time.time() - t0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, "JPEG", quality=95)
    print(f"[phase_a_persona] [{account_id}]   portrait rendered in {elapsed:.1f}s "
          f"-> {out_path}")

    return {
        "local_path": str(out_path),
        "fal_url": None,
        "seed": seed,
        "request_id": f"local-{uuid.uuid4().hex[:8]}",
        "elapsed_seconds": elapsed,
        "endpoint": ENDPOINT_LABEL,
        "cost_usd": COST_PER_IMAGE_USD,
        "params_used": {
            "width": width, "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "max_sequence_length": max_sequence_length,
            "seed": seed,
        },
    }


# ──────────────────────────────────────────────────────────────────────────
# Standalone test — full Phase A for ONE account: Opus prompt -> FLUX portrait.
# Renders a real face you can look at. ~$0.10 Opus + GPU time.
#   cd /workspace/alluvi-pipeline
#   python supabase_pipeline/phase_a_persona.py 1
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import supabase_db
    import phase_a_prompt_builder

    accounts = supabase_db.get_all_accounts()
    if not accounts:
        print("[phase_a_persona] no accounts in tiktok_accounts")
        sys.exit(1)

    want = int(sys.argv[1]) if len(sys.argv) > 1 else accounts[0]["id"]
    account = next((a for a in accounts if a["id"] == want), None)
    if account is None:
        print(f"[phase_a_persona] no account with id={want}")
        sys.exit(1)

    print(f"\n=== Phase A test: id={account['id']} {account['tiktok_id']} "
          f"({account['gender']}, {account['country']}, age {account['age']}) ===\n")

    # 1. Opus -> appearance + portrait prompt
    envelope = phase_a_prompt_builder.build_appearance_prompt(account)
    portrait_prompt = envelope["portrait_prompt"]
    print("\n--- portrait_prompt ---")
    print(portrait_prompt)

    # 2. FLUX -> portrait
    safe_id = account["tiktok_id"].lstrip("@").replace("/", "_")
    out_path = REPO_ROOT / "outputs" / "_phaseA_test" / f"{safe_id}_portrait.jpg"

    print("\n[phase_a_persona] loading FLUX.1-dev ...")
    pipe = load_pipeline()
    try:
        meta = generate(pipe, portrait_prompt, out_path, account_id=account["tiktok_id"])
    finally:
        unload_pipeline(pipe)

    print(f"\n[phase_a_persona] DONE — portrait at:\n  {meta['local_path']}")
    print("Open that file to eyeball the reference face.")
