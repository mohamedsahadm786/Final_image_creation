"""
video_pipeline/step_5_video_wan.py — Stage 5: image-to-video via Wan 2.2 on
ComfyUI #1 (:8188). Loads the captured wan_api.json and overrides only the
dynamic fields, then POSTs to /prompt and waits (this is the slow stage).

Audio-FIRST: num_frames is derived from the TTS audio duration, snapped UP to
Wan's 4n+1, capped at 81 (~5s). Video length >= audio length, so LatentSync
trims rather than loops.

Public surface:
  load_pipeline() -> dict handle
  generate(handle, image_path, out_path, *, motion_prompt, audio_duration_seconds
           or num_frames, negative_prompt, seed, ...) -> dict
  unload_pipeline(handle) -> None
"""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
import socket
import time
import uuid
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "127.0.0.1")
COMFYUI_PORT = int(os.environ.get("COMFYUI_PORT", "8188"))
SERVER_ADDR = f"{COMFYUI_HOST}:{COMFYUI_PORT}"
COMFYUI_INPUT_DIR = Path(os.environ.get(
    "COMFYUI_STAGE2_INPUT_DIR", "/workspace/comfyui-stage2/ComfyUI/input"))
COMFYUI_OUTPUT_DIR = Path(os.environ.get(
    "COMFYUI_STAGE2_OUTPUT_DIR", "/workspace/comfyui-stage2/ComfyUI/output"))

WORKFLOW_PATH = Path(os.environ.get(
    "WAN_WORKFLOW_PATH", str(Path(__file__).parent / "workflows" / "wan_api.json")))

ENDPOINT_LABEL = "local/comfyui-wan2.2-i2v"
MODEL_NAME = "Wan2.2-I2V-A14B"
WORKFLOW_TEMPLATE = "wan_i2v_alluvi_ladder_v3"
COST_PER_CALL_USD = 0.0
JOB_TIMEOUT_S = 1800
VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".mov")

DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 1344
DEFAULT_FPS = 16
FRAME_CAP = 81

# The proven Wan negative from the tested workflow (standard Wan zh negative +
# face/teeth/mouth terms). Script negatives are prepended to this.
BASE_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走, "
    "face distortion, facial asymmetry, extra teeth, deformed teeth, oversized teeth, "
    "large front teeth, mouth warping, distorted mouth, malformed mouth, teeth artifacts, "
    "ugly teeth"
)


def _server_is_up() -> bool:
    try:
        with socket.create_connection((COMFYUI_HOST, COMFYUI_PORT), timeout=2):
            pass
        with urllib.request.urlopen(f"http://{SERVER_ADDR}/system_stats", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def load_pipeline() -> dict:
    if not _server_is_up():
        raise RuntimeError(
            f"ComfyUI #1 not reachable on {SERVER_ADDR}. "
            f"Start it with: bash /workspace/start_pipeline.sh")
    if not WORKFLOW_PATH.exists():
        raise FileNotFoundError(f"Wan workflow not found at {WORKFLOW_PATH}")
    print(f"[step_5_wan] ComfyUI #1 reachable on {SERVER_ADDR}")
    return {"client_id": uuid.uuid4().hex}


def unload_pipeline(handle: dict | None) -> None:
    return


def frames_for_duration(sec: float, fps: int = DEFAULT_FPS, cap: int = FRAME_CAP) -> int:
    """Smallest 4n+1 frame count whose length >= the audio (capped ~5s)."""
    raw = math.ceil(sec * fps)
    n = math.ceil((raw - 1) / 4)
    return max(5, min(4 * n + 1, cap))


def _find(graph: dict, class_type: str, title_contains: str | None = None):
    for nid, node in graph.items():
        if node.get("class_type") == class_type:
            if title_contains is None or \
               title_contains.lower() in node.get("_meta", {}).get("title", "").lower():
                return nid
    return None


def _find_seed_sampler(graph: dict):
    for nid, node in graph.items():
        if node.get("class_type") == "KSamplerAdvanced" \
           and node.get("inputs", {}).get("add_noise") == "enable":
            return nid
    return None


def _stage_input_image(image_path: Path) -> str:
    src = Path(image_path)
    if not src.exists():
        raise FileNotFoundError(f"scene image not found: {src}")
    COMFYUI_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    dst_name = f"wanin_{uuid.uuid4().hex[:8]}{src.suffix}"
    shutil.copy(src, COMFYUI_INPUT_DIR / dst_name)
    return dst_name


def _queue(graph: dict, client_id: str) -> str:
    payload = json.dumps({"prompt": graph, "client_id": client_id}).encode("utf-8")
    req = urllib.request.Request(f"http://{SERVER_ADDR}/prompt", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["prompt_id"]


def _wait(prompt_id: str, timeout: int = JOB_TIMEOUT_S) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(
                    f"http://{SERVER_ADDR}/history/{prompt_id}", timeout=10) as r:
                data = json.loads(r.read())
            if prompt_id in data:
                return data[prompt_id]
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError(f"Wan job {prompt_id} did not finish in {timeout}s")


def _find_output_file(entry: dict) -> tuple[str, str]:
    found = []
    for node_out in entry.get("outputs", {}).values():
        for val in node_out.values():
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict) and item.get("filename"):
                        found.append((item["filename"], item.get("subfolder", "")))
    if not found:
        raise RuntimeError("no video output found in ComfyUI history entry")
    for fn, sub in found:
        if fn.lower().endswith(VIDEO_EXTS):
            return fn, sub
    return found[0]


def generate(handle: dict, image_path, out_path, *,
             motion_prompt: str,
             audio_duration_seconds: float | None = None,
             num_frames: int | None = None,
             negative_prompt: str | None = None,
             width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT,
             fps: int = DEFAULT_FPS, seed: int | None = None,
             filename_prefix: str = "video/alluvi_wan",
             scene_id: str = "?") -> dict:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if num_frames is None:
        if audio_duration_seconds is None:
            raise ValueError("pass num_frames or audio_duration_seconds")
        num_frames = frames_for_duration(audio_duration_seconds, fps)
    if seed is None:
        seed = uuid.uuid4().int % (2 ** 48)
    neg = (negative_prompt.strip() + ", " + BASE_NEGATIVE) if negative_prompt else BASE_NEGATIVE

    graph = copy.deepcopy(json.loads(WORKFLOW_PATH.read_text(encoding="utf-8")))
    n_img = _find(graph, "LoadImage")
    n_pos = _find(graph, "CLIPTextEncode", "positive")
    n_neg = _find(graph, "CLIPTextEncode", "negative")
    n_wan = _find(graph, "WanImageToVideo")
    n_save = _find(graph, "SaveVideo") or _find(graph, "CreateVideo")
    n_seed = _find_seed_sampler(graph)
    for label, nid in [("LoadImage", n_img), ("positive", n_pos), ("negative", n_neg),
                       ("WanImageToVideo", n_wan), ("seed sampler", n_seed)]:
        if nid is None:
            raise RuntimeError(f"could not locate {label} node in wan_api.json")

    image_name = _stage_input_image(image_path)
    graph[n_img]["inputs"]["image"] = image_name
    graph[n_pos]["inputs"]["text"] = motion_prompt
    graph[n_neg]["inputs"]["text"] = neg
    graph[n_wan]["inputs"]["length"] = int(num_frames)   # literal int (bypasses math chain)
    graph[n_wan]["inputs"]["width"] = int(width)
    graph[n_wan]["inputs"]["height"] = int(height)
    graph[n_seed]["inputs"]["noise_seed"] = int(seed)
    if n_save and "filename_prefix" in graph[n_save].get("inputs", {}):
        graph[n_save]["inputs"]["filename_prefix"] = filename_prefix

    audit = {"endpoint": ENDPOINT_LABEL, "server": SERVER_ADDR,
             "workflow_template": WORKFLOW_TEMPLATE,
             "arguments": {"image": image_name, "motion_prompt": motion_prompt,
                           "negative_prompt": neg, "num_frames": num_frames,
                           "width": width, "height": height, "fps": fps, "seed": seed}}
    (out_path.parent / f"{out_path.stem}_request.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[step_5_wan] [{scene_id}] queueing Wan ({num_frames} frames @ {fps}fps "
          f"= {num_frames/fps:.2f}s, {width}x{height}, seed={seed}) — this is slow")
    t0 = time.time()
    prompt_id = _queue(graph, handle["client_id"])
    entry = _wait(prompt_id)
    filename, subfolder = _find_output_file(entry)
    elapsed = time.time() - t0

    src = COMFYUI_OUTPUT_DIR / subfolder / filename
    if not src.exists():
        raise FileNotFoundError(
            f"ComfyUI reported {src} but it's not on disk "
            f"(check COMFYUI_STAGE2_OUTPUT_DIR; currently {COMFYUI_OUTPUT_DIR})")
    final = out_path.parent / f"{out_path.stem}{src.suffix}"
    shutil.copy(src, final)
    print(f"[step_5_wan] [{scene_id}]   video in {elapsed:.1f}s -> {final}")

    return {
        "local_path": str(final),
        "num_frames": num_frames,
        "fps": fps,
        "seed": seed,
        "request_id": prompt_id,
        "elapsed_seconds": elapsed,
        "endpoint": ENDPOINT_LABEL,
        "model_name": MODEL_NAME,
        "workflow_template": WORKFLOW_TEMPLATE,
        "comfyui_server": f"http://{SERVER_ADDR}",
        "negative_prompt": neg,
        "cost_usd": COST_PER_CALL_USD,
        "params": {"width": width, "height": height, "fps": fps,
                   "num_frames": num_frames, "seed": seed},
    }


if __name__ == "__main__":
    import sys
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from video_pipeline import video_db

    rows = video_db.get_outputs_needing_video(limit=1)
    if not rows:
        print("no finished scene images need a video (check outputs table)")
        sys.exit(1)
    row = rows[0]
    img = row["final_image_local_path"]
    sid = row["scenario_id"]
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 4.66

    motion = ("The woman blinks naturally and breathes softly with a gentle, calm "
              "expression, mouth closed and relaxed, a faint smile forming. She holds "
              "the product box completely still and steady. An extremely slow, almost "
              "imperceptible camera push-in. Face remains stable, identity preserved, "
              "product label stable, cinematic realism.")

    print(f"\n=== Wan test ===\nimage: {img}\nscene: {sid}\nduration: {dur}s "
          f"-> {frames_for_duration(dur)} frames\n")
    h = load_pipeline()
    meta = generate(h, img, Path("/workspace/alluvi-pipeline/outputs/_wan_test/test.mp4"),
                    motion_prompt=motion, audio_duration_seconds=dur, scene_id=sid)
    print("\n--- RESULT ---")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nWatch: {meta['local_path']}")
