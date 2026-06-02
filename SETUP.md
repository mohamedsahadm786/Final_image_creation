# Alluvi Pipeline — New-Server Setup / Migration Guide

This repo holds ONLY code, prompts, configs, workflows. No models, no venvs, no ComfyUI.
Everything else lives as siblings under /workspace and is rebuilt from upstream (below).
Exact dependency sets: setup/locks/. Exact model files on disk: setup/MODELS_INVENTORY.txt.

## 0. Architecture
- Two-phase GPU use: images (~95GB) and Wan video (~65GB) never co-resident.
- ComfyUI #1 :8188 = Qwen image-edit (Stage 2) + Wan 2.2 video.
- ComfyUI #2 :8189 = F5-TTS + LatentSync.

## 1. System prerequisites
- Ubuntu, NVIDIA driver + CUDA 12.8, large-VRAM GPU.
- sudo apt install -y git git-lfs ffmpeg build-essential python3.11 python3.11-venv
- Persistent disk mounted at /workspace.

## 2. Clone third-party repos (pinned to in-use commits)
# ComfyUI #1 (Qwen + Wan)
git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/comfyui-stage2/ComfyUI
cd /workspace/comfyui-stage2/ComfyUI && git checkout 72e3f608
git clone https://github.com/lrzjason/Comfyui-QwenEditUtils.git custom_nodes/Comfyui-QwenEditUtils
cd custom_nodes/Comfyui-QwenEditUtils && git checkout cdd4d02

# ComfyUI #2 (TTS + LatentSync)
git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/comfyui-tts
cd /workspace/comfyui-tts && git checkout d80fcafe
git clone https://github.com/ShmuelRonen/ComfyUI-LatentSyncWrapper.git custom_nodes/ComfyUI-LatentSyncWrapper && (cd custom_nodes/ComfyUI-LatentSyncWrapper && git checkout 360d528)
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git custom_nodes/ComfyUI-VideoHelperSuite && (cd custom_nodes/ComfyUI-VideoHelperSuite && git checkout 4ee72c0)
git clone https://github.com/diodiogod/TTS-Audio-Suite.git custom_nodes/TTS-Audio-Suite && (cd custom_nodes/TTS-Audio-Suite && git checkout 9699393)

# PuLID + ai-toolkit
git clone https://github.com/ToTheBeginning/PuLID.git /workspace/PuLID
git clone https://github.com/ostris/ai-toolkit.git /workspace/ai-toolkit
# (To take latest instead of these commits, skip the git checkout lines and re-verify.)

## 3. Build venvs (Python 3.11.10)
# Alluvi venv (inside ai-toolkit; runs the pipeline)
cd /workspace/ai-toolkit && python3.11 -m venv venv && source venv/bin/activate && pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r /workspace/alluvi-pipeline/setup/locks/alluvi_venv_freeze.txt
# (alluvi-pipeline/requirements.txt predates Supabase; the freeze above is authoritative.)

# ComfyUI #1 venv
cd /workspace/comfyui-stage2/ComfyUI && python3.11 -m venv venv && source venv/bin/activate && pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -r custom_nodes/Comfyui-QwenEditUtils/requirements.txt
# reference: setup/locks/comfyui_stage2_venv_freeze.txt

# ComfyUI #2 venv
cd /workspace/comfyui-tts && python3.11 -m venv venv && source venv/bin/activate && pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
for n in ComfyUI-LatentSyncWrapper ComfyUI-VideoHelperSuite TTS-Audio-Suite; do pip install -r custom_nodes/$n/requirements.txt; done
# reference: setup/locks/comfyui_tts_venv_freeze.txt

# NOTE: torch lines assume CUDA 12.8. Different CUDA on the new box = change cu128. This is the
# one thing most likely to need per-machine adjustment.

## 4. Download models (manual) — exact files & sizes in setup/MODELS_INVENTORY.txt
# Target dirs:
#   /workspace/models/ : FLUX.1-dev, Qwen-Image-Edit-2511 (diffusers), FLUX.1-Kontext-dev,
#                        PuLID weights, insightface/models/antelopev2
#                        (FLUX + Kontext are GATED on HuggingFace: accept license + huggingface-cli login)
#   /workspace/comfyui-stage2/ComfyUI/models/diffusion_models/ :
#       qwen_image_edit_2511_bf16.safetensors
#       wan2.2_i2v_high_noise_14B_fp16.safetensors
#       wan2.2_i2v_low_noise_14B_fp16.safetensors
#   .../models/text_encoders/ : qwen_2.5_vl_7b_fp8_scaled.safetensors , umt5_xxl (variant per inventory)
#   .../models/vae/           : qwen_image_vae.safetensors , wan_2.1_vae.safetensors
#   /workspace/comfyui-tts/   : F5TTS_v1_Base + LatentSync 1.6 checkpoints (land in custom-node
#                               checkpoint folders; see inventory for exact paths)
#
# Verified public Wan URLs (no login):
wget -c "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
wget -c "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"
wget -c "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors"
# For every other model, match filename+path in MODELS_INVENTORY.txt to its HuggingFace source.

## 5. This repo + secrets
git clone https://github.com/mohamedsahadm786/Final_image_creation.git /workspace/alluvi-pipeline
cd /workspace/alluvi-pipeline && cp .env.example .env
# fill .env: SUPABASE_URL, SUPABASE_SECRET_KEY, ANTHROPIC_API_KEY (+ INSIGHTFACE_HOME, HF_HOME if used)

## 6. Run
# Terminal A (:8188): cd /workspace/comfyui-stage2/ComfyUI && source venv/bin/activate && python main.py --listen 0.0.0.0 --port 8188
# Terminal B (:8189): cd /workspace/comfyui-tts && source venv/bin/activate && python main.py --listen 0.0.0.0 --port 8189
# Terminal C: cd /workspace/alluvi-pipeline && source /workspace/ai-toolkit/venv/bin/activate
#             python master.py --accounts emma.callahan --num 1 --yes
# (Long runs in tmux.)
