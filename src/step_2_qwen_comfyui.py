"""
src/step_2_qwen_comfyui.py — Stage 2 product compositing via ComfyUI backend.

DROP-IN REPLACEMENT for src/step_2_qwen_edit.py. Same public surface, same
return-dict contract, same audit JSON.

WORKFLOW SELECTION (the comeback path):
  Two workflow JSON templates live in comfyui_workflows/:
    - qwen_edit_2511_product.json          (BASIC TextEncodeQwenImageEditPlus)
    - qwen_edit_2511_product_advance.json  (ADVANCE node — product on the
                                            not_resize path for pixel fidelity)
  Which one is used is controlled by the env var ALLUVI_COMFY_WORKFLOW:
    unset / "basic"   -> qwen_edit_2511_product.json
    "advance"         -> qwen_edit_2511_product_advance.json
  To revert to the known-good basic workflow: `unset ALLUVI_COMFY_WORKFLOW`.
  No code edit needed to switch.

WHY COMFYUI:
  diffusers QwenImageEditPlusPipeline regenerates the product instead of
  pixel-copying it. ComfyUI's Advance node routes the product image through
  the VAE path at full resolution (not_resize_image2) so its text/graphics
  are preserved, while the persona drives the VL semantic path.

PUBLIC SURFACE (identical to step_2_qwen_edit):
  load_pipeline()           -> ComfyUIHandle
  generate(pipeline, ...)   -> dict
  unload_pipeline(handle)   -> None

RETURN DICT (identical contract):
  {local_path, fal_url(None), seed, request_id("local-<uuid>"),
   elapsed_seconds, endpoint, cost_usd(0.0)}
"""

import io
import json
import os
import shutil
import socket
import subprocess
import time
import uuid
import urllib.request
import urllib.parse
from pathlib import Path

import websocket  # websocket-client

# ── Project paths ────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCT_IMAGE_PATH = REPO_ROOT / "assets" / "product.jpg"
WORKFLOW_DIR = REPO_ROOT / "comfyui_workflows"

# Workflow selection via env var (the comeback switch)
_WF_NAME = os.environ.get("ALLUVI_COMFY_WORKFLOW", "basic").strip().lower()
if _WF_NAME == "advance":
    WORKFLOW_TEMPLATE_PATH = WORKFLOW_DIR / "qwen_edit_2511_product_advance.json"
else:
    WORKFLOW_TEMPLATE_PATH = WORKFLOW_DIR / "qwen_edit_2511_product.json"

# ── ComfyUI install + server config ──────────────────────────────────────
COMFYUI_DIR = Path(os.environ.get(
    "COMFYUI_DIR", "/workspace/comfyui-stage2/ComfyUI"))
COMFYUI_PYTHON = Path(os.environ.get(
    "COMFYUI_PYTHON", "/workspace/comfyui-stage2/ComfyUI/venv/bin/python"))
COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "127.0.0.1")
COMFYUI_PORT = int(os.environ.get("COMFYUI_PORT", "8188"))
SERVER_ADDR = f"{COMFYUI_HOST}:{COMFYUI_PORT}"

COMFYUI_INPUT_DIR = COMFYUI_DIR / "input"

ENDPOINT_LABEL = "local/comfyui-qwen-image-edit-2511"
COST_PER_IMAGE_USD = 0.0

# ── Inference defaults ───────────────────────────────────────────────────
DEFAULT_NUM_STEPS = 30
DEFAULT_CFG = 3.0
DEFAULT_NEGATIVE_PROMPT = " "

SERVER_BOOT_TIMEOUT_S = 240
JOB_TIMEOUT_S = 600


# ══════════════════════════════════════════════════════════════════════════
# Server lifecycle
# ══════════════════════════════════════════════════════════════════════════

class ComfyUIHandle:
    def __init__(self, process: subprocess.Popen | None, started_by_us: bool):
        self.process = process
        self.started_by_us = started_by_us
        self.client_id = uuid.uuid4().hex


def _server_is_up() -> bool:
    try:
        with socket.create_connection((COMFYUI_HOST, COMFYUI_PORT), timeout=2):
            pass
    except OSError:
        return False
    try:
        with urllib.request.urlopen(
                f"http://{SERVER_ADDR}/system_stats", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def load_pipeline(model_path: str | None = None) -> ComfyUIHandle:
    """Ensure the ComfyUI server is running; return a handle.
    `model_path` is accepted for signature-compatibility but unused."""
    print(f"[step_2_comfyui] workflow template: {WORKFLOW_TEMPLATE_PATH.name} "
          f"(ALLUVI_COMFY_WORKFLOW={_WF_NAME})")
    if _server_is_up():
        print(f"[step_2_comfyui] ComfyUI already running on {SERVER_ADDR}")
        return ComfyUIHandle(process=None, started_by_us=False)

    if not COMFYUI_PYTHON.exists():
        raise FileNotFoundError(f"ComfyUI python not found at {COMFYUI_PYTHON}")
    if not (COMFYUI_DIR / "main.py").exists():
        raise FileNotFoundError(f"ComfyUI main.py not found in {COMFYUI_DIR}")

    print(f"[step_2_comfyui] starting ComfyUI server on {SERVER_ADDR} ...")
    log_path = COMFYUI_DIR / "comfyui_server.log"
    log_fh = open(log_path, "a", encoding="utf-8")
    process = subprocess.Popen(
        [str(COMFYUI_PYTHON), "main.py",
         "--listen", COMFYUI_HOST, "--port", str(COMFYUI_PORT)],
        cwd=str(COMFYUI_DIR), stdout=log_fh, stderr=subprocess.STDOUT,
    )
    t0 = time.time()
    while time.time() - t0 < SERVER_BOOT_TIMEOUT_S:
        if _server_is_up():
            print(f"[step_2_comfyui]   server up in {time.time()-t0:.1f}s")
            return ComfyUIHandle(process=process, started_by_us=True)
        if process.poll() is not None:
            raise RuntimeError(f"ComfyUI exited during boot — see {log_path}")
        time.sleep(2)
    process.terminate()
    raise TimeoutError(f"ComfyUI did not boot in {SERVER_BOOT_TIMEOUT_S}s")


def unload_pipeline(handle: ComfyUIHandle) -> None:
    if handle is None:
        return
    if handle.started_by_us and handle.process is not None:
        print("[step_2_comfyui] stopping ComfyUI server (started by pipeline)")
        try:
            handle.process.terminate()
            handle.process.wait(timeout=20)
        except Exception:
            try:
                handle.process.kill()
            except Exception:
                pass
    else:
        print("[step_2_comfyui] leaving ComfyUI server running")


# ══════════════════════════════════════════════════════════════════════════
# Workflow templating + API
# ══════════════════════════════════════════════════════════════════════════

def _parse_image_size(value) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        w, h = value.get("width"), value.get("height")
        if w and h:
            return int(w), int(h)
    presets = {
        "square_hd": (1024, 1024), "square": (512, 512),
        "portrait_4_3": (768, 1024), "portrait_16_9": (576, 1024),
        "landscape_4_3": (1024, 768), "landscape_16_9": (1024, 576),
    }
    if isinstance(value, str) and value in presets:
        return presets[value]
    return None


def _build_workflow(prompt: str, negative: str,
                    persona_name: str, product_name: str,
                    width: int, height: int, steps: int,
                    cfg: float, seed: int) -> dict:
    """Load the JSON template and substitute placeholders.

    Numeric placeholders are QUOTED in the template (e.g. "__SEED__") so the
    file stays valid JSON; we replace the quoted form with a bare number.
    String placeholders are set after json.loads() so prompt text with
    quotes/newlines is handled safely.
    """
    raw = WORKFLOW_TEMPLATE_PATH.read_text(encoding="utf-8")
    # numeric — replace the quoted placeholder so result is a bare int/float
    raw = raw.replace('"__WIDTH__"', str(width))
    raw = raw.replace('"__HEIGHT__"', str(height))
    raw = raw.replace('"__SEED__"', str(seed))
    raw = raw.replace('"__STEPS__"', str(steps))
    raw = raw.replace('"__CFG__"', str(cfg))
    wf = json.loads(raw)
    # string placeholders — set as proper JSON string values
    if "4" in wf:
        wf["4"]["inputs"]["image"] = persona_name
    if "5" in wf:
        wf["5"]["inputs"]["image"] = product_name
    if "6" in wf:
        wf["6"]["inputs"]["prompt"] = prompt
    if "7" in wf:
        wf["7"]["inputs"]["prompt"] = negative
    wf.pop("_comment", None)
    return wf


def _queue_prompt(workflow: dict, client_id: str) -> str:
    payload = json.dumps({"prompt": workflow, "client_id": client_id}).encode()
    req = urllib.request.Request(f"http://{SERVER_ADDR}/prompt", data=payload)
    resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
    if "error" in resp:
        raise RuntimeError(
            f"ComfyUI rejected workflow: {json.dumps(resp, indent=2)}")
    return resp["prompt_id"]


def _wait_for_done(prompt_id: str, client_id: str) -> None:
    ws = websocket.WebSocket()
    ws.connect(f"ws://{SERVER_ADDR}/ws?clientId={client_id}", timeout=30)
    ws.settimeout(JOB_TIMEOUT_S)
    try:
        while True:
            msg = ws.recv()
            if not isinstance(msg, str):
                continue
            data = json.loads(msg)
            if data.get("type") == "executing":
                d = data["data"]
                if d.get("node") is None and d.get("prompt_id") == prompt_id:
                    return
            elif data.get("type") == "execution_error":
                raise RuntimeError(
                    f"ComfyUI execution error: {json.dumps(data['data'])}")
    finally:
        ws.close()


def _fetch_result_bytes(prompt_id: str) -> bytes:
    with urllib.request.urlopen(
            f"http://{SERVER_ADDR}/history/{prompt_id}", timeout=30) as r:
        hist = json.loads(r.read())
    if prompt_id not in hist:
        raise RuntimeError(f"prompt_id {prompt_id} not in /history")
    outputs = hist[prompt_id].get("outputs", {})
    for node_out in outputs.values():
        if "images" in node_out:
            for img in node_out["images"]:
                params = urllib.parse.urlencode(img)
                with urllib.request.urlopen(
                        f"http://{SERVER_ADDR}/view?{params}", timeout=30) as r:
                    return r.read()
    raise RuntimeError("no output image in ComfyUI /history result")


# ══════════════════════════════════════════════════════════════════════════
# Public generate()
# ══════════════════════════════════════════════════════════════════════════

def generate(
    pipeline: ComfyUIHandle,
    step_1_local_path: str,
    step_2_prompt: str,
    fal_qwen_params: dict,
    out_path: Path,
    scenario_id: str = "unknown",
) -> dict:
    """Run Qwen-Image-Edit-2511 product compositing through ComfyUI.
    Signature identical to step_2_qwen_edit.generate()."""
    if not PRODUCT_IMAGE_PATH.exists():
        raise FileNotFoundError(f"product.jpg not found at {PRODUCT_IMAGE_PATH}")
    if not Path(step_1_local_path).exists():
        raise FileNotFoundError(f"Step 1 image not found at {step_1_local_path}")
    if not WORKFLOW_TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"workflow template not found: {WORKFLOW_TEMPLATE_PATH}")
    if not _server_is_up():
        raise RuntimeError(
            f"ComfyUI server not reachable on {SERVER_ADDR} — "
            f"call load_pipeline() first")

    client_id = pipeline.client_id if pipeline else uuid.uuid4().hex

    num_steps = int(fal_qwen_params.get("num_inference_steps", DEFAULT_NUM_STEPS))
    cfg = float(fal_qwen_params.get("true_cfg_scale",
                fal_qwen_params.get("cfg", DEFAULT_CFG)))
    negative = fal_qwen_params.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT
    seed = fal_qwen_params.get("seed")
    if seed is None:
        seed = uuid.uuid4().int % (2 ** 31)
    seed = int(seed)

    size = _parse_image_size(fal_qwen_params.get("image_size"))
    if size:
        width, height = size
    else:
        from PIL import Image
        im = Image.open(step_1_local_path)
        width, height = im.width, im.height

    # stage input images into ComfyUI/input/
    COMFYUI_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{scenario_id}_{uuid.uuid4().hex[:8]}".replace("#", "_").replace("/", "_")
    persona_name = f"alluvi_persona_{tag}.jpg"
    product_name = f"alluvi_product_{tag}.jpg"
    shutil.copy(step_1_local_path, COMFYUI_INPUT_DIR / persona_name)
    shutil.copy(PRODUCT_IMAGE_PATH, COMFYUI_INPUT_DIR / product_name)

    workflow = _build_workflow(
        prompt=step_2_prompt, negative=negative,
        persona_name=persona_name, product_name=product_name,
        width=width, height=height, steps=num_steps,
        cfg=cfg, seed=seed,
    )

    resolved_args = {
        "prompt": step_2_prompt,
        "negative_prompt": negative,
        "image_urls": [str(Path(step_1_local_path).resolve()),
                       str(PRODUCT_IMAGE_PATH.resolve())],
        "num_inference_steps": num_steps,
        "cfg": cfg,
        "width": width, "height": height,
        "seed": seed,
        "backend": "comfyui",
        "workflow_template": WORKFLOW_TEMPLATE_PATH.name,
        "comfyui_server": SERVER_ADDR,
    }
    audit_path = out_path.parent / f"{out_path.stem}_request.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps({"endpoint": ENDPOINT_LABEL, "arguments": resolved_args},
                   indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"[step_2_comfyui] [{scenario_id}]   request audit -> {audit_path.name}")
    print(f"[step_2_comfyui] [{scenario_id}] submitting "
          f"(wf={WORKFLOW_TEMPLATE_PATH.name}, steps={num_steps}, cfg={cfg}, "
          f"size={width}x{height}, seed={seed})")

    t0 = time.time()
    prompt_id = _queue_prompt(workflow, client_id)
    _wait_for_done(prompt_id, client_id)
    image_bytes = _fetch_result_bytes(prompt_id)
    elapsed = time.time() - t0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.save(out_path, "JPEG", quality=95)

    for nm in (persona_name, product_name):
        try:
            (COMFYUI_INPUT_DIR / nm).unlink(missing_ok=True)
        except Exception:
            pass

    print(f"[step_2_comfyui] [{scenario_id}]   composited in {elapsed:.1f}s")

    return {
        "local_path": str(out_path),
        "fal_url": None,
        "seed": seed,
        "request_id": f"local-{uuid.uuid4().hex[:8]}",
        "elapsed_seconds": elapsed,
        "endpoint": ENDPOINT_LABEL,
        "cost_usd": COST_PER_IMAGE_USD,
    }