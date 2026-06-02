"""
src/pulid_inference/pulid_pipeline.py — wraps guozinan/PuLID's FluxGenerator.

PuLID's app_flux.py defines a FluxGenerator class that loads FLUX.1-dev +
PuLID + InsightFace and runs identity-preserving image generation. We
import it as-is (no fork, no vendoring) and wrap it with:

  - Symlinks for FLUX.1-dev weights into /workspace/PuLID/models/
    (PuLID's flux/util.py uses CWD-relative 'models/flux1-dev.safetensors' paths)
  - chdir to /workspace/PuLID during init + inference (same reason)
  - PIL ↔ numpy conversion for the persona reference image
  - Clean Python kwargs interface (no SimpleNamespace, no -1 seed convention)

FIRST-RUN DOWNLOADS (one-time, cached after — set HF_HOME=/workspace/hf_cache
in .env so these persist across pod restarts):
  - antelopev2 face model → /workspace/PuLID/models/antelopev2/ (~400 MB)
    (NOT symlinked to /workspace/models/insightface/ — different HF mirror,
    hash mismatch would trigger re-download anyway)
  - T5-XXL text encoder (xlabs-ai/xflux_text_encoders) (~10 GB)
  - CLIP-L (openai/clip-vit-large-patch14) (~1.7 GB)
  - facexlib detection/parsing models (~50 MB)

VRAM: ~28 GB at bf16 (FLUX-dev transformer + VAE + T5 + CLIP + PuLID encoder
+ EVA-CLIP face encoder + face_helper). InsightFace ONNX models run via
onnxruntime-gpu CUDA provider — separate allocation, freed when the
generator is deleted.
"""

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image


PULID_REPO_ROOT = Path(os.environ.get("PULID_REPO_ROOT", "/workspace/PuLID"))
PULID_WEIGHTS_DIR = Path(os.environ.get("PULID_WEIGHTS_PATH", "/workspace/models/PuLID"))
FLUX_DEV_DIR = Path(os.environ.get("FLUX_DEV_MODEL_PATH", "/workspace/models/FLUX.1-dev"))
PULID_VERSION = os.environ.get("PULID_VERSION", "v0.9.1")


def _setup_pulid_workspace() -> None:
    """
    Idempotent: create the /workspace/PuLID/models/ symlinks that PuLID's
    CWD-relative path resolution expects.

    Symlinks:
      models/flux1-dev.safetensors                → /workspace/models/FLUX.1-dev/flux1-dev.safetensors
      models/ae.safetensors                        → /workspace/models/FLUX.1-dev/ae.safetensors
      models/pulid_flux_<version>.safetensors      → /workspace/models/PuLID/pulid_flux_<version>.safetensors

    Does NOT symlink antelopev2 — PuLID downloads it itself on first run.
    """
    if not PULID_REPO_ROOT.exists():
        raise FileNotFoundError(
            f"PuLID code repo not found at {PULID_REPO_ROOT}. "
            f"Clone: git clone https://github.com/ToTheBeginning/PuLID.git {PULID_REPO_ROOT}"
        )

    pulid_models_dir = PULID_REPO_ROOT / "models"
    pulid_models_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        pulid_models_dir / "flux1-dev.safetensors":
            FLUX_DEV_DIR / "flux1-dev.safetensors",
        pulid_models_dir / "ae.safetensors":
            FLUX_DEV_DIR / "ae.safetensors",
        pulid_models_dir / f"pulid_flux_{PULID_VERSION}.safetensors":
            PULID_WEIGHTS_DIR / f"pulid_flux_{PULID_VERSION}.safetensors",
    }

    for link, target in targets.items():
        if not target.exists():
            raise FileNotFoundError(
                f"PuLID dependency missing on disk: {target}\n"
                f"(needed as symlink target for {link})"
            )
        if link.is_symlink():
            try:
                if Path(os.readlink(link)).resolve() == target.resolve():
                    continue  # already correct
            except OSError:
                pass
            link.unlink()
        elif link.exists():
            # Real file at this path — don't clobber, let user manage
            print(f"[pulid_pipeline] note: {link} exists as a real file, not symlinking")
            continue
        link.symlink_to(target)
        print(f"[pulid_pipeline] symlink: {link.name} → {target}")


class PulidWrapper:
    """
    Thin wrapper around guozinan/PuLID's FluxGenerator.

    Hides PuLID's CWD-relative path handling and converts the Gradio-style
    numpy interface to a clean kwargs interface.
    """

    def __init__(
        self,
        model_name: str = "flux-dev",
        version: str = PULID_VERSION,
        fp8: bool = False,
        onnx_provider: str = "gpu",
    ):
        _setup_pulid_workspace()

        if str(PULID_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(PULID_REPO_ROOT))

        # Import AFTER sys.path is set. FluxGenerator is at the repo root in
        # app_flux.py and imports gradio at module level, so gradio must be
        # installed (we never use the UI — just need the import to succeed).
        from app_flux import FluxGenerator

        args = SimpleNamespace(
            fp8=fp8,
            onnx_provider=onnx_provider,
            pretrained_model=str(
                PULID_WEIGHTS_DIR / f"pulid_flux_{version}.safetensors"
            ),
            version=version,
        )

        # PuLID's flux/util.py uses CWD-relative paths so we chdir during init.
        original_cwd = Path.cwd()
        try:
            os.chdir(PULID_REPO_ROOT)
            self.generator = FluxGenerator(
                model_name=model_name,
                device="cuda",
                offload=False,             # 96 GB Blackwell — no offload needed
                aggressive_offload=False,
                args=args,
            )
        finally:
            os.chdir(original_cwd)

    def to(self, device: str):
        """
        Bulk-move the underlying models. Used by vram_utils.unload_pipeline()
        to free VRAM before deletion.
        """
        try:
            self.generator.model = self.generator.model.to(device)
            self.generator.ae = self.generator.ae.to(device)
            self.generator.t5 = self.generator.t5.to(device)
            self.generator.clip = self.generator.clip.to(device)
            self.generator.pulid_model.components_to_device(torch.device(device))
        except Exception as e:
            # Partial move is acceptable — the unload helper does del + gc + empty_cache
            # right after, which will reclaim the rest.
            print(f"[pulid_pipeline] partial .to({device}) move: {e}")
        return self

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        persona_image_path: str,
        *,
        width: int = 768,
        height: int = 1344,
        num_inference_steps: int = 30,
        guidance_scale: float = 3.5,
        id_weight: float = 1.0,
        true_cfg: float = 1.5,
        negative_prompt: str = "",
        max_sequence_length: int = 512,
        seed: int | None = None,
        start_step: int = 0,
        timestep_to_start_cfg: int = 1,
    ) -> tuple[Image.Image, int]:
        """
        Run PuLID inference. Returns (PIL output image, used seed).
        """
        # PIL → numpy RGB uint8 (the format PuLIDPipeline.get_id_embedding expects).
        id_image_pil = Image.open(persona_image_path).convert("RGB")
        id_image_np = np.array(id_image_pil)

        # PuLID seed convention: -1 means random (handled internally by FluxGenerator).
        seed_arg = int(seed) if seed is not None else -1

        original_cwd = Path.cwd()
        try:
            os.chdir(PULID_REPO_ROOT)
            img, used_seed_str, _debug_imgs = self.generator.generate_image(
                width=width,
                height=height,
                num_steps=num_inference_steps,
                start_step=start_step,
                guidance=guidance_scale,
                seed=seed_arg,
                prompt=prompt,
                id_image=id_image_np,
                id_weight=id_weight,
                neg_prompt=negative_prompt,
                true_cfg=true_cfg,
                timestep_to_start_cfg=timestep_to_start_cfg,
                max_sequence_length=max_sequence_length,
            )
        finally:
            os.chdir(original_cwd)

        return img, int(used_seed_str)