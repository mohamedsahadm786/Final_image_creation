"""
ollama_flow/preflight_ollama.py — preflight checks for the Ollama flow (LOCAL pipeline).

Run automatically by run_ollama.py / run_batch_ollama.py / run_ollama_resident.py,
or directly via: `python preflight_ollama.py`

CHECKS (8 total):
  1. .env vars (OLLAMA_HOST, OLLAMA_MODEL, INSIGHTFACE_HOME, HF_HOME — NO FAL_KEY anymore)
  2. Ollama server reachability + qwen2.5:7b model availability
  3. Shared asset files (persona, product)
  4. Brand + scenarios + prompts
  5. scenarios.yaml structure
  6. Python deps (diffusers, transformers, accelerate, peft, insightface, onnxruntime, etc.)
  7. Diffusers pipeline classes can be imported
  8. Project module imports (parent's src.* + ollama_src.*)
  9. Model paths on disk (FLUX-dev, Qwen-Edit, Kontext, PuLID, antelopev2)
"""

import os
import sys
import importlib
from pathlib import Path
from typing import Tuple

OLLAMA_FLOW_ROOT = Path(__file__).resolve().parent
PARENT_REPO_ROOT = OLLAMA_FLOW_ROOT.parent

# Put both on sys.path
sys.path.insert(0, str(OLLAMA_FLOW_ROOT))
sys.path.insert(0, str(PARENT_REPO_ROOT))

# Load env from ollama_flow/.env (NOT parent .env)
try:
    from dotenv import load_dotenv
    load_dotenv(OLLAMA_FLOW_ROOT / ".env")
except ImportError:
    pass

MIN_DEPS = [
    ("diffusers",        "diffusers",        ">=0.32  -- HuggingFace diffusion pipelines"),
    ("transformers",     "transformers",     "         -- HuggingFace transformers"),
    ("accelerate",       "accelerate",       "         -- model device placement"),
    ("peft",             "peft",             "         -- LoRA / adapter support"),
    ("sentencepiece",    "sentencepiece",    "         -- T5 tokenizer dependency"),
    ("safetensors",      "safetensors",      "         -- model weight format"),
    ("einops",           "einops",           "         -- tensor manipulation"),
    ("insightface",      "insightface",      "         -- PuLID face embeddings"),
    ("onnxruntime",      "onnxruntime",      "         -- InsightFace ONNX backend"),
    ("httpx",            "httpx",            "         -- HTTP — used by Ollama client"),
    ("dotenv",           "python-dotenv",    "         -- env loader"),
    ("yaml",             "PyYAML",           "         -- config + scenarios"),
    ("sqlite3",          "stdlib",           "         -- database"),
    ("PIL",              "Pillow",           "         -- image I/O"),
]

DIFFUSERS_PIPELINES = [
    ("diffusers", "FluxPipeline",            "FLUX-dev base (used by PuLID Stage 1 fallback)"),
    ("diffusers", "FluxKontextPipeline",     "FLUX.1-Kontext-dev (Stage 3 realism)"),
    ("diffusers", "QwenImageEditPlusPipeline", "Qwen-Image-Edit-2511 (Stage 2)"),
]

MODEL_PATHS = [
    ("/workspace/models/FLUX.1-dev",                         "FLUX-dev base weights"),
    ("/workspace/models/Qwen-Image-Edit-2511",               "Qwen-Image-Edit-2511 weights"),
    ("/workspace/models/FLUX.1-Kontext-dev",                 "FLUX.1-Kontext-dev weights"),
    ("/workspace/models/PuLID",                              "PuLID adapter weights"),
    ("/workspace/models/insightface/models/antelopev2",      "InsightFace antelopev2 (face embeddings)"),
]


def _fmt_size(p: Path) -> str:
    try:
        if p.is_dir():
            total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            return f"{total / (1024**3):.1f} GB"
        return f"{p.stat().st_size / 1024:.1f} KB"
    except Exception:
        return "?"


def run_preflight(verbose: bool = True) -> Tuple[list, list]:
    """Returns (errors, warnings). Empty errors == preflight passes."""
    errors: list[str] = []
    warnings: list[str] = []

    def pr(msg):
        if verbose:
            print(msg)

    pr("=" * 72)
    pr(" ALLUVI — OLLAMA FLOW PREFLIGHT (local pipeline)")
    pr("=" * 72)

    # ─── 1. .env vars ────────────────────────────────────────────────────
    pr("[1/9] Checking ollama_flow/.env vars...")
    ollama_host = os.getenv("OLLAMA_HOST", "")
    ollama_model = os.getenv("OLLAMA_MODEL", "")
    insightface_home = os.getenv("INSIGHTFACE_HOME", "")
    hf_home = os.getenv("HF_HOME", "")

    if ollama_host:
        pr(f"  OK     OLLAMA_HOST       = {ollama_host}")
    else:
        warnings.append("OLLAMA_HOST not set — default http://localhost:11434 will be used")
        pr(f"  WARN   OLLAMA_HOST not set — default http://localhost:11434 will be used")

    if ollama_model:
        pr(f"  OK     OLLAMA_MODEL      = {ollama_model}")
    else:
        warnings.append("OLLAMA_MODEL not set — default qwen2.5:7b will be used")
        pr(f"  WARN   OLLAMA_MODEL not set — default qwen2.5:7b will be used")

    if insightface_home:
        pr(f"  OK     INSIGHTFACE_HOME  = {insightface_home}")
    else:
        errors.append("INSIGHTFACE_HOME not set in ollama_flow/.env")
        pr(f"  X      INSIGHTFACE_HOME not set — needed by PuLID Stage 1")

    if hf_home:
        pr(f"  OK     HF_HOME           = {hf_home}")
    else:
        warnings.append("HF_HOME not set — HuggingFace downloads will land in ~/.cache/huggingface")
        pr(f"  WARN   HF_HOME not set — HF downloads will go to default cache (not persisted on volume)")

    # ─── 2. Ollama reachability ──────────────────────────────────────────
    pr("[2/9] Checking Ollama server reachability + model availability...")
    try:
        import httpx
        url = (ollama_host or "http://localhost:11434").rstrip("/") + "/api/tags"
        resp = httpx.get(url, timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            model_names = [m.get("name", "") for m in data.get("models", [])]
            target_model = ollama_model or "qwen2.5:7b"
            if any(target_model in n for n in model_names):
                pr(f"  OK     Ollama reachable, model '{target_model}' is pulled")
            else:
                errors.append(f"Ollama is up but model '{target_model}' is not pulled")
                pr(f"  X      Ollama is up but '{target_model}' is not pulled. Run: ollama pull {target_model}")
        else:
            errors.append(f"Ollama returned HTTP {resp.status_code} from {url}")
            pr(f"  X      Ollama returned HTTP {resp.status_code} from {url}")
    except Exception as e:
        errors.append(f"Ollama not reachable: {e}")
        pr(f"  X      Ollama not reachable at {ollama_host}: {type(e).__name__}: {e}")
        pr(f"         (Start it with: export OLLAMA_MODELS=/workspace/ollama && ollama serve &)")

    # ─── 3. Shared asset files ───────────────────────────────────────────
    pr("[3/9] Checking shared asset files (from parent repo)...")
    asset_files = [
        ("../assets/persona.jpg",  "Step 1 PuLID reference (full body)"),
        ("../assets/persona.yaml", "Persona identity descriptor"),
        ("../assets/product.jpg",  "Step 2 product reference"),
        ("../assets/product.yaml", "Product packaging descriptor"),
    ]
    for rel_path, label in asset_files:
        p = (OLLAMA_FLOW_ROOT / rel_path).resolve()
        if p.exists():
            pr(f"  OK     {rel_path}  ({_fmt_size(p)})  -- {label}")
        else:
            errors.append(f"missing {p}")
            pr(f"  X      {rel_path}  -- MISSING ({label})")

    # ─── 4. Brand + scenarios + prompts ──────────────────────────────────
    pr("[4/9] Checking brand, scenarios, prompts (from parent repo)...")
    other_files = [
        ("../brand/brand.yaml",                     None),
        ("../brand/do_dont.md",                     None),
        ("../scenarios/scenarios.yaml",             "30 hand-curated scenarios"),
        ("../prompts/master_prompt_step1.md",       "Step 1 PuLID system prompt"),
        ("../prompts/master_prompt_step2_qwen.md",  "Step 2 Qwen-tuned system prompt"),
    ]
    for rel_path, label in other_files:
        p = (OLLAMA_FLOW_ROOT / rel_path).resolve()
        if p.exists():
            label_str = f"  -- {label}" if label else ""
            pr(f"  OK     {rel_path}  ({p.stat().st_size:,} bytes){label_str}")
        else:
            errors.append(f"missing {p}")
            pr(f"  X      {rel_path}  -- MISSING")

    # ─── 5. scenarios.yaml structure ─────────────────────────────────────
    pr("[5/9] Validating scenarios.yaml structure...")
    try:
        from src import scenario_loader
        scenarios = scenario_loader.load_scenarios()
        n = len(scenarios)
        if n == 0:
            errors.append("scenarios.yaml: no scenarios loaded")
            pr(f"  X      scenarios.yaml: no scenarios loaded")
        else:
            pr(f"  OK     {n} scenarios loaded and validated")
            from collections import Counter
            arche = Counter(s.get("archetype", "?") for s in scenarios)
            diff = Counter(s.get("difficulty", "?") for s in scenarios)
            cat = Counter(s.get("category", "?") for s in scenarios)
            pr(f"  INFO   by archetype : {dict(arche)}")
            pr(f"  INFO   by difficulty: {dict(diff)}")
            pr(f"  INFO   by category  : {dict(cat)}")
    except Exception as e:
        errors.append(f"scenarios.yaml validation failed: {e}")
        pr(f"  X      scenarios.yaml validation failed: {type(e).__name__}: {e}")

    # ─── 6. Python deps ──────────────────────────────────────────────────
    pr("[6/9] Checking Python dependencies...")
    for module, package, note in MIN_DEPS:
        try:
            mod = importlib.import_module(module)
            ver = getattr(mod, "__version__", "?")
            pr(f"  OK     {package:<14} {ver:<14}{note}")
        except ImportError as e:
            errors.append(f"missing python dep: {package}")
            pr(f"  X      {package:<14} MISSING        -- pip install {package}")

    # ─── 7. Diffusers pipeline classes ───────────────────────────────────
    pr("[7/9] Checking diffusers pipeline classes...")
    for module, classname, label in DIFFUSERS_PIPELINES:
        try:
            mod = importlib.import_module(module)
            cls = getattr(mod, classname, None)
            if cls is None:
                errors.append(f"{module}.{classname} not found — diffusers may be outdated")
                pr(f"  X      {classname:<28} NOT FOUND -- {label}")
            else:
                pr(f"  OK     {classname:<28} -- {label}")
        except Exception as e:
            errors.append(f"{module}.{classname} import failed: {e}")
            pr(f"  X      {classname:<28} import failed: {e}")

    # ─── 8. Project module imports ───────────────────────────────────────
    pr("[8/9] Checking project module imports...")
    modules = [
        ("parent: src.db",                                 "src.db"),
        ("parent: src.scenario_loader",                    "src.scenario_loader"),
        ("parent: src.step_1_pulid",                       "src.step_1_pulid"),
        ("parent: src.step_2_qwen_edit",                   "src.step_2_qwen_edit"),
        ("parent: src.step_3_realism",                     "src.step_3_realism"),
        ("parent: src.trace_html",                         "src.trace_html"),
        ("parent: src.overview_html",                      "src.overview_html"),
        ("parent: src.vram_utils",                         "src.vram_utils"),
        ("parent: src.pulid_inference.pulid_pipeline",     "src.pulid_inference.pulid_pipeline"),
        ("ollama: ollama_src.ollama_client",               "ollama_src.ollama_client"),
        ("ollama: ollama_src.step_1_prompt_builder_ollama","ollama_src.step_1_prompt_builder_ollama"),
        ("ollama: ollama_src.step_2_prompt_builder_ollama","ollama_src.step_2_prompt_builder_ollama"),
    ]
    for label, modname in modules:
        try:
            importlib.import_module(modname)
            pr(f"  OK     {label}")
        except Exception as e:
            errors.append(f"{modname} import failed: {e}")
            pr(f"  X      {label}  -- import failed: {type(e).__name__}: {e}")

    # ─── 9. Model paths on disk ──────────────────────────────────────────
    pr("[9/9] Checking model paths on disk...")
    for path, label in MODEL_PATHS:
        p = Path(path)
        if p.exists() and any(p.iterdir()):
            pr(f"  OK     {path}  ({_fmt_size(p)})  -- {label}")
        elif p.exists():
            errors.append(f"{path} exists but is empty")
            pr(f"  X      {path}  -- EXISTS BUT EMPTY ({label})")
        else:
            errors.append(f"{path} missing")
            pr(f"  X      {path}  -- MISSING ({label})")

    # ─── Final summary ───────────────────────────────────────────────────
    pr("=" * 72)
    if errors:
        pr(f" OLLAMA-FLOW PREFLIGHT FAILED — {len(errors)} error(s)")
        pr("=" * 72)
        pr("Errors to fix before running:")
        for e in errors:
            pr(f"  X {e}")
    else:
        pr(f" OLLAMA-FLOW PREFLIGHT PASSED" + (f" — {len(warnings)} warning(s)" if warnings else ""))
        pr("=" * 72)

    return errors, warnings


if __name__ == "__main__":
    errors, _warnings = run_preflight(verbose=True)
    sys.exit(0 if not errors else 1)