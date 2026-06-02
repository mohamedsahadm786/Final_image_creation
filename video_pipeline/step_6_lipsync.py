"""
video_pipeline/step_6_lipsync.py — Stage 6: lip-sync via LatentSync 1.6 on
ComfyUI #2 (:8189). Takes the silent Wan clip + the TTS audio and repaints the
mouth to match the voice. Loads the captured latentsync_api.json and overrides
only the video input, audio input, and seed.

VideoLengthAdjuster(loop_to_audio) matches the clip to the audio length, so the
final mp4 runs exactly as long as the voiceover.

Public surface:
  load_pipeline() -> dict handle
  generate(handle, video_path, audio_path, out_path, *, seed, ...) -> dict
  unload_pipeline(handle) -> None
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import socket
import time
import uuid
import urllib.request
from pathlib import Path

COMFYUI_TTS_HOST = os.environ.get("COMFYUI_TTS_HOST", "127.0.0.1")
COMFYUI_TTS_PORT = int(os.environ.get("COMFYUI_TTS_PORT", "8189"))
SERVER_ADDR = f"{COMFYUI_TTS_HOST}:{COMFYUI_TTS_PORT}"
COMFYUI_TTS_INPUT_DIR = Path(os.environ.get(
    "COMFYUI_TTS_INPUT_DIR", "/workspace/comfyui-tts/input"))
COMFYUI_TTS_OUTPUT_DIR = Path(os.environ.get(
    "COMFYUI_TTS_OUTPUT_DIR", "/workspace/comfyui-tts/output"))

WORKFLOW_PATH = Path(os.environ.get(
    "LATENTSYNC_WORKFLOW_PATH",
    str(Path(__file__).parent / "workflows" / "latentsync_api.json")))

ENDPOINT_LABEL = "local/comfyui-latentsync"
MODEL_NAME = "LatentSync-1.6"
WORKFLOW_TEMPLATE = "latentsync_alluvi_test_v2"
COST_PER_CALL_USD = 0.0
JOB_TIMEOUT_S = int(os.environ.get("LATENTSYNC_TIMEOUT_S", "3600"))
VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".mov")


def _server_is_up() -> bool:
    try:
        with socket.create_connection((COMFYUI_TTS_HOST, COMFYUI_TTS_PORT), timeout=2):
            pass
        with urllib.request.urlopen(f"http://{SERVER_ADDR}/system_stats", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def load_pipeline() -> dict:
    if not _server_is_up():
        raise RuntimeError(
            f"ComfyUI #2 not reachable on {SERVER_ADDR}. "
            f"Start it with: bash /workspace/start_pipeline.sh")
    if not WORKFLOW_PATH.exists():
        raise FileNotFoundError(f"LatentSync workflow not found at {WORKFLOW_PATH}")
    print(f"[step_6_lipsync] ComfyUI #2 reachable on {SERVER_ADDR}")
    return {"client_id": uuid.uuid4().hex}


def unload_pipeline(handle: dict | None) -> None:
    return


def _find(graph: dict, class_type: str):
    for nid, node in graph.items():
        if node.get("class_type") == class_type:
            return nid
    return None


def _stage_input(path, prefix: str) -> str:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"{prefix} input not found: {src}")
    COMFYUI_TTS_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{prefix}_{uuid.uuid4().hex[:8]}{src.suffix}"
    shutil.copy(src, COMFYUI_TTS_INPUT_DIR / name)
    return name


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
    raise TimeoutError(f"LatentSync job {prompt_id} did not finish in {timeout}s")


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


def generate(handle: dict, video_path, audio_path, out_path, *,
             seed: int | None = None,
             filename_prefix: str = "alluvi_lipsync",
             scene_id: str = "?",
             lips_expression: float | None = None,
             inference_steps: int | None = None) -> dict:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if seed is None:
        seed = uuid.uuid4().int % (2 ** 31)

    graph = copy.deepcopy(json.loads(WORKFLOW_PATH.read_text(encoding="utf-8")))
    n_video = _find(graph, "VHS_LoadVideo")
    n_audio = _find(graph, "LoadAudio")
    n_sync = _find(graph, "LatentSyncNode")
    n_save = _find(graph, "SaveVideo")
    for label, nid in [("VHS_LoadVideo", n_video), ("LoadAudio", n_audio),
                       ("LatentSyncNode", n_sync)]:
        if nid is None:
            raise RuntimeError(f"could not locate {label} node in latentsync_api.json")

    video_name = _stage_input(video_path, "lipvid")
    audio_name = _stage_input(audio_path, "lipaud")
    graph[n_video]["inputs"]["video"] = video_name
    graph[n_audio]["inputs"]["audio"] = audio_name
    graph[n_audio]["inputs"].pop("audioUI", None)
    graph[n_sync]["inputs"]["seed"] = int(seed)
    if lips_expression is not None:
        graph[n_sync]["inputs"]["lips_expression"] = float(lips_expression)
    if inference_steps is not None:
        graph[n_sync]["inputs"]["inference_steps"] = int(inference_steps)
    if lips_expression is not None:
        graph[n_sync]["inputs"]["lips_expression"] = float(lips_expression)
    if inference_steps is not None:
        graph[n_sync]["inputs"]["inference_steps"] = int(inference_steps)
    if n_save and "filename_prefix" in graph[n_save].get("inputs", {}):
        graph[n_save]["inputs"]["filename_prefix"] = filename_prefix

    audit = {"endpoint": ENDPOINT_LABEL, "server": SERVER_ADDR,
             "workflow_template": WORKFLOW_TEMPLATE,
             "arguments": {"video": video_name, "audio": audio_name, "seed": seed}}
    (out_path.parent / f"{out_path.stem}_request.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[step_6_lipsync] [{scene_id}] queueing LatentSync (seed={seed})")
    t0 = time.time()
    prompt_id = _queue(graph, handle["client_id"])
    entry = _wait(prompt_id)
    filename, subfolder = _find_output_file(entry)
    elapsed = time.time() - t0

    src = COMFYUI_TTS_OUTPUT_DIR / subfolder / filename
    if not src.exists():
        raise FileNotFoundError(
            f"ComfyUI reported {src} but it's not on disk "
            f"(check COMFYUI_TTS_OUTPUT_DIR; currently {COMFYUI_TTS_OUTPUT_DIR})")
    final = out_path.parent / f"{out_path.stem}{src.suffix}"
    shutil.copy(src, final)
    print(f"[step_6_lipsync] [{scene_id}]   lip-synced video in {elapsed:.1f}s -> {final}")

    return {
        "local_path": str(final),
        "seed": seed,
        "request_id": prompt_id,
        "elapsed_seconds": elapsed,
        "endpoint": ENDPOINT_LABEL,
        "model_name": MODEL_NAME,
        "workflow_template": WORKFLOW_TEMPLATE,
        "comfyui_server": f"http://{SERVER_ADDR}",
        "cost_usd": COST_PER_CALL_USD,
        "params": {"seed": seed},
    }


if __name__ == "__main__":
    import sys
    video = (sys.argv[1] if len(sys.argv) > 1
             else "/workspace/alluvi-pipeline/outputs/_wan_test/test.mp4")
    audio = (sys.argv[2] if len(sys.argv) > 2
             else "/workspace/alluvi-pipeline/outputs/_tts_test/test_voice.flac")
    out = Path("/workspace/alluvi-pipeline/outputs/_lipsync_test/test_final.mp4")
    print(f"\n=== LatentSync test ===\nvideo: {video}\naudio: {audio}\n")
    h = load_pipeline()
    meta = generate(h, video, audio, out, scene_id="lipsync_test")
    print("\n--- RESULT ---")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nWatch: {meta['local_path']}")


# --- AUTO-TRIM: drop LatentSync silent_padding tail by trimming to audio length ---
try:
    try:
        from video_pipeline.trim_to_audio import trim_video_to_audio as _trim_to_audio
    except Exception:
        from trim_to_audio import trim_video_to_audio as _trim_to_audio
except Exception:
    _trim_to_audio = None

_latentsync_generate_raw = generate
def generate(*args, **kwargs):
    res = _latentsync_generate_raw(*args, **kwargs)
    try:
        audio_path = kwargs.get("audio_path")
        if audio_path is None and len(args) > 2:
            audio_path = args[2]
        final_path = res.get("local_path") if isinstance(res, dict) else None
        if _trim_to_audio and audio_path and final_path:
            _trim_to_audio(final_path, audio_path)
            print(f"[step_6_lipsync] trimmed to audio length: {final_path}")
    except Exception as e:
        print(f"[step_6_lipsync] trim skipped (kept untrimmed): {e}")
    return res
# --- end AUTO-TRIM ---
