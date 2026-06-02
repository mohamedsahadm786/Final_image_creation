"""
video_pipeline/step_4_tts_f5.py — Stage 4: voiceover via F5-TTS on ComfyUI #2 (:8189).

Audio-FIRST: the returned audio_duration_seconds is what drives the Wan frame
count in Stage 5. Mirrors the HTTP pattern of src/step_2_qwen_comfyui.py:
POST the workflow graph to /prompt, poll /history, read the saved file from
ComfyUI #2's output dir, copy it to out_path, measure duration with ffprobe.

Public surface:
  load_pipeline() -> dict handle      (verifies :8189 is reachable)
  generate(handle, dialogue, out_path, *, narrator_voice, seed, ...) -> dict
  unload_pipeline(handle) -> None     (no-op; the server stays up)

Return dict:
  {local_path, audio_duration_seconds, seed, request_id, elapsed_seconds,
   endpoint, model_name, narrator_voice, workflow_template, comfyui_server,
   cost_usd, params}
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
import uuid
import urllib.request
from pathlib import Path

COMFYUI_TTS_HOST = os.environ.get("COMFYUI_TTS_HOST", "127.0.0.1")
COMFYUI_TTS_PORT = int(os.environ.get("COMFYUI_TTS_PORT", "8189"))
SERVER_ADDR = f"{COMFYUI_TTS_HOST}:{COMFYUI_TTS_PORT}"
COMFYUI_TTS_OUTPUT_DIR = Path(os.environ.get(
    "COMFYUI_TTS_OUTPUT_DIR", "/workspace/comfyui-tts/output"))

ENDPOINT_LABEL = "local/comfyui-f5-tts"
MODEL_NAME = "F5TTS_v1_Base"
WORKFLOW_TEMPLATE = "tts_f5_alluvi_test_v2"
COST_PER_CALL_USD = 0.0
DEFAULT_NARRATOR_VOICE = "voices_examples/female/female_02.wav"
JOB_TIMEOUT_S = 300

# F5 engine params, copied verbatim from the tested workflow (node 1).
DEFAULT_ENGINE = {
    "language": "F5TTS_v1_Base",
    "device": "auto",
    "temperature": 0.8,
    "speed": 1.0,
    "target_rms": 0.1,
    "cross_fade_duration": 0.15,
    "nfe_step": 32,
    "cfg_strength": 2.0,
}


def _server_is_up() -> bool:
    try:
        with socket.create_connection((COMFYUI_TTS_HOST, COMFYUI_TTS_PORT), timeout=2):
            pass
        with urllib.request.urlopen(f"http://{SERVER_ADDR}/system_stats", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def load_pipeline() -> dict:
    """Verify ComfyUI #2 is reachable (it's started by start_pipeline.sh)."""
    if not _server_is_up():
        raise RuntimeError(
            f"ComfyUI #2 not reachable on {SERVER_ADDR}. "
            f"Start it with: bash /workspace/start_pipeline.sh")
    print(f"[step_4_tts] ComfyUI #2 reachable on {SERVER_ADDR}")
    return {"client_id": uuid.uuid4().hex}


def unload_pipeline(handle: dict | None) -> None:
    return  # server is shared / long-lived; nothing to unload


def _build_workflow(dialogue: str, narrator_voice: str, seed: int,
                    filename_prefix: str, engine: dict) -> dict:
    return {
        "1": {"class_type": "F5TTSEngineNode", "inputs": dict(engine)},
        "2": {"class_type": "UnifiedTTSTextNode", "inputs": {
            "text": dialogue,
            "narrator_voice": narrator_voice,
            "seed": int(seed),
            "enable_chunking": True,
            "max_chars_per_chunk": 400,
            "chunk_combination_method": "auto",
            "silence_between_chunks_ms": 100,
            "enable_audio_cache": True,
            "batch_size": 0,
            "TTS_engine": ["1", 0],
        }},
        "3": {"class_type": "SaveAudio", "inputs": {
            "filename_prefix": filename_prefix,
            "audio": ["2", 0],
        }},
    }


def _queue(graph: dict, client_id: str) -> str:
    payload = json.dumps({"prompt": graph, "client_id": client_id}).encode("utf-8")
    req = urllib.request.Request(
        f"http://{SERVER_ADDR}/prompt", data=payload,
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
        time.sleep(1.5)
    raise TimeoutError(f"TTS job {prompt_id} did not finish in {timeout}s")


def _find_output_file(entry: dict) -> tuple[str, str]:
    for node_out in entry.get("outputs", {}).values():
        for val in node_out.values():
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict) and item.get("filename"):
                        return item["filename"], item.get("subfolder", "")
    raise RuntimeError("no audio output found in ComfyUI history entry")


def _audio_duration(path: Path) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
    return float(out.decode().strip())


def generate(handle: dict, dialogue: str, out_path: Path, *,
             narrator_voice: str = DEFAULT_NARRATOR_VOICE,
             seed: int = 42,
             engine_overrides: dict | None = None,
             filename_prefix: str = "alluvi_pipe",
             scene_id: str = "?") -> dict:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    engine = dict(DEFAULT_ENGINE)
    if engine_overrides:
        engine.update(engine_overrides)

    graph = _build_workflow(dialogue, narrator_voice, seed, filename_prefix, engine)

    # audit sidecar (for media_generations)
    audit = {"endpoint": ENDPOINT_LABEL, "server": SERVER_ADDR,
             "workflow_template": WORKFLOW_TEMPLATE,
             "arguments": {"text": dialogue, "narrator_voice": narrator_voice,
                           "seed": seed, **engine}}
    (out_path.parent / f"{out_path.stem}_request.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[step_4_tts] [{scene_id}] queueing F5-TTS "
          f"(voice={narrator_voice}, seed={seed})")
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
    duration = _audio_duration(final)
    print(f"[step_4_tts] [{scene_id}]   audio {duration:.2f}s in {elapsed:.1f}s "
          f"-> {final}")

    return {
        "local_path": str(final),
        "audio_duration_seconds": duration,
        "seed": seed,
        "request_id": prompt_id,
        "elapsed_seconds": elapsed,
        "endpoint": ENDPOINT_LABEL,
        "model_name": MODEL_NAME,
        "narrator_voice": narrator_voice,
        "workflow_template": WORKFLOW_TEMPLATE,
        "comfyui_server": f"http://{SERVER_ADDR}",
        "cost_usd": COST_PER_CALL_USD,
        "params": {**engine, "text": dialogue, "seed": seed,
                   "narrator_voice": narrator_voice},
    }


if __name__ == "__main__":
    import sys
    line = (sys.argv[1] if len(sys.argv) > 1
            else "Honestly, showing up for myself like this has been everything for me lately.")
    out = Path("/workspace/alluvi-pipeline/outputs/_tts_test/test_voice")
    print(f"\n=== F5-TTS test ===\ntext: {line}\n")
    h = load_pipeline()
    meta = generate(h, line, out, scene_id="tts_test")
    print("\n--- RESULT ---")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nListen to: {meta['local_path']}  (duration {meta['audio_duration_seconds']:.2f}s)")
