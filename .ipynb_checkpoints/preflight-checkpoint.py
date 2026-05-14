"""
preflight.py — Pre-run validation for the Alluvi LOCAL Image Generation pipeline.

Validates that every required file exists, every API key is set, every Python
dependency is installed, every Python module is importable, AND every local
model weight is on disk. Catches all common setup errors BEFORE you spend any
LLM credits or GPU time on a real run.

Standalone:
    python preflight.py
    exits with code 0 on success, 1 on any error.

Auto-invoked:
    run_batch.py calls run_preflight() before its cost-confirmation prompt.

Changes from the fal-version preflight:
  - Drops FAL_KEY check (no fal usage)
  - Drops fal_client dependency check
  - Adds diffusers / transformers / accelerate / insightface / onnxruntime-gpu / torch checks
  - Adds [8/8] local model paths existence check
  - Updates the final 'ready to run' commands to point at orchestration/per_scenario/
"""

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


# Local model paths — must exist for inference to start
EXPECTED_MODEL_PATHS = [
    ("FLUX.1-dev",            "/workspace/models/FLUX.1-dev"),
    ("Qwen-Image-Edit-2511",  "/workspace/models/Qwen-Image-Edit-2511"),
    ("FLUX.1-Kontext-dev",    "/workspace/models/FLUX.1-Kontext-dev"),
    ("PuLID weights",         "/workspace/models/PuLID"),
    ("InsightFace antelopev2","/workspace/models/insightface/models/antelopev2"),
]


# ──────────────────────────────────────────────────────────────────────────
# Individual checks — each returns lines and mutates errors/warnings
# ──────────────────────────────────────────────────────────────────────────

def _check_env_keys(errors: list[str]) -> list[str]:
    lines: list[str] = []
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
        # Only Anthropic key required now — Opus prompts + Sonnet QC.
        for key_name in ("ANTHROPIC_API_KEY",):
            if os.getenv(key_name):
                value = os.getenv(key_name)
                masked = value[:8] + "..." + value[-4:] if len(value) > 16 else "***"
                lines.append(f"  OK     {key_name} = {masked}")
            else:
                errors.append(f".env missing {key_name}")
                lines.append(f"  MISSING {key_name}")

        # INSIGHTFACE_HOME is optional but recommended — pin it to avoid surprises
        ih = os.getenv("INSIGHTFACE_HOME")
        if ih:
            lines.append(f"  OK     INSIGHTFACE_HOME = {ih}")
        else:
            lines.append(
                "  INFO   INSIGHTFACE_HOME unset — set in .env "
                "(recommended: /workspace/models/insightface)"
            )
    except ImportError:
        errors.append("python-dotenv not installed: run `pip install -r requirements.txt`")
        lines.append("  ERROR  python-dotenv not installed")
    return lines


def _check_asset_files(errors: list[str]) -> list[str]:
    lines: list[str] = []
    asset_files = [
        ("assets/persona.jpg", "Step 1 PuLID reference (full body)"),
        ("assets/persona.yaml", "Persona identity descriptor"),
        ("assets/product.jpg", "Step 2 product reference"),
        ("assets/product.yaml", "Product packaging descriptor"),
    ]
    for fpath, role in asset_files:
        p = REPO_ROOT / fpath
        if p.exists():
            size_kb = p.stat().st_size / 1024
            lines.append(f"  OK     {fpath:<32} ({size_kb:>6.1f} KB)  -- {role}")
        else:
            errors.append(f"missing asset file: {fpath} ({role})")
            lines.append(f"  MISSING {fpath}  -- {role}")
    return lines


def _check_brand_files(errors: list[str], warnings: list[str]) -> list[str]:
    lines: list[str] = []
    brand_files = [
        "brand/brand.yaml",
        "brand/do_dont.md",
    ]
    for fpath in brand_files:
        p = REPO_ROOT / fpath
        if p.exists() and p.stat().st_size > 100:
            lines.append(f"  OK     {fpath} ({p.stat().st_size:,} bytes)")
        elif p.exists():
            warnings.append(f"{fpath} exists but is suspiciously small")
            lines.append(f"  WARN   {fpath} (only {p.stat().st_size} bytes)")
        else:
            errors.append(f"missing brand file: {fpath}")
            lines.append(f"  MISSING {fpath}")
    return lines


def _check_scenarios_and_prompts(errors: list[str], warnings: list[str]) -> list[str]:
    lines: list[str] = []
    required_files = [
        ("scenarios/scenarios.yaml", "30 hand-curated scenarios"),
        ("prompts/master_prompt_step1.md", "Step 1 PuLID system prompt"),
        ("prompts/master_prompt_step2_qwen.md", "Step 2 Qwen-tuned system prompt"),
    ]
    for fpath, role in required_files:
        p = REPO_ROOT / fpath
        if p.exists() and p.stat().st_size > 1000:
            lines.append(f"  OK     {fpath:<45} ({p.stat().st_size:>6,} bytes)  -- {role}")
        elif p.exists():
            warnings.append(f"{fpath} exists but is small ({p.stat().st_size} bytes)")
            lines.append(f"  WARN   {fpath} (only {p.stat().st_size} bytes)")
        else:
            errors.append(f"missing required file: {fpath}")
            lines.append(f"  MISSING {fpath}")
    return lines


def _check_scenarios_yaml_structure(errors: list[str]) -> list[str]:
    lines: list[str] = []
    scenarios_path = REPO_ROOT / "scenarios" / "scenarios.yaml"
    if not scenarios_path.exists():
        lines.append("  SKIP   scenarios.yaml missing (already reported)")
        return lines

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    try:
        from src.scenario_loader import load_scenarios

        scenarios = load_scenarios()
        lines.append(f"  OK     {len(scenarios)} scenarios loaded and validated")

        from collections import Counter
        by_archetype = Counter(s["archetype"] for s in scenarios)
        by_difficulty = Counter(s["difficulty"] for s in scenarios)
        by_category = Counter(s["category"] for s in scenarios)

        lines.append(f"  INFO   by archetype : {dict(by_archetype)}")
        lines.append(f"  INFO   by difficulty: {dict(by_difficulty)}")
        lines.append(f"  INFO   by category  : {dict(by_category)}")

    except ImportError as e:
        errors.append(f"could not import scenario_loader: {e}")
        lines.append(f"  ERROR  import failed: {e}")
    except FileNotFoundError as e:
        errors.append(str(e))
        lines.append(f"  ERROR  {e}")
    except ValueError as e:
        errors.append("scenarios.yaml validation failed (see lines below)")
        lines.append(f"  ERROR  scenarios.yaml validation FAILED:")
        for line in str(e).splitlines():
            lines.append(f"         {line}")
    except Exception as e:
        errors.append(f"scenarios.yaml unexpected error: {type(e).__name__}: {e}")
        lines.append(f"  ERROR  {type(e).__name__}: {e}")

    return lines


def _check_python_dependencies(errors: list[str], warnings: list[str]) -> list[str]:
    lines: list[str] = []
    deps = [
        ("torch", "PyTorch (Blackwell CUDA stack)"),
        ("diffusers", "FluxPipeline / FluxKontextPipeline / QwenImageEditPlusPipeline"),
        ("transformers", "text encoders for FLUX + Qwen"),
        ("accelerate", "model loading + device management"),
        ("safetensors", "weights format"),
        ("insightface", "face embedding for PuLID"),
        ("onnxruntime", "GPU runtime for InsightFace (provided by onnxruntime-gpu)"),
        ("anthropic", "Claude Opus 4.7 + Sonnet 4.6 SDK"),
        ("httpx", "HTTP client"),
        ("dotenv", "via python-dotenv"),
        ("yaml", "via PyYAML"),
        ("PIL", "via Pillow"),
    ]
    for module_name, role in deps:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "?")
            lines.append(f"  OK     {module_name:<14} {version:<14}  -- {role}")
        except ImportError:
            errors.append(
                f"missing dependency: {module_name} -- run `pip install -r requirements.txt`"
            )
            lines.append(f"  MISSING {module_name}  -- {role}")

    # Verify CUDA is actually usable
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            dev = torch.cuda.get_device_name(0)
            lines.append(f"  OK     CUDA available — {n} device(s), GPU 0 = {dev}")
        else:
            errors.append("torch.cuda.is_available() == False")
            lines.append("  ERROR  CUDA not available — local inference will not work")
    except Exception as e:
        warnings.append(f"could not probe CUDA: {e}")

    # Verify the specific diffusers pipeline classes import
    pipeline_classes = [
        "FluxPipeline",
        "FluxKontextPipeline",
        "QwenImageEditPlusPipeline",
    ]
    for cls_name in pipeline_classes:
        try:
            mod = __import__("diffusers", fromlist=[cls_name])
            getattr(mod, cls_name)
            lines.append(f"  OK     diffusers.{cls_name}")
        except (ImportError, AttributeError) as e:
            errors.append(f"diffusers.{cls_name} not available: {e}")
            lines.append(f"  MISSING diffusers.{cls_name}  -- {e}")

    # sqlite3 is stdlib
    try:
        import sqlite3
        lines.append(
            f"  OK     sqlite3        {sqlite3.sqlite_version:<14}  -- stdlib (database)"
        )
    except ImportError:
        errors.append("sqlite3 not available (Python stdlib problem)")
        lines.append("  ERROR  sqlite3 not available")

    return lines


def _check_project_imports(errors: list[str]) -> list[str]:
    lines: list[str] = []
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    src_modules = [
        "src.db",
        "src.scenario_loader",
        "src.json_utils",
        "src.qc_validator",
        "src.vram_utils",
        "src.step_1_prompt_builder",
        "src.step_1_pulid",
        "src.step_2_prompt_builder",
        "src.step_2_qwen_edit",
        "src.step_3_realism",
        "src.trace_html",
        "src.overview_html",
    ]
    for mod_name in src_modules:
        try:
            __import__(mod_name)
            lines.append(f"  OK     {mod_name}")
        except ModuleNotFoundError as e:
            if "src" in str(e):
                errors.append(f"missing project file: {mod_name.replace('.', '/')}.py")
                lines.append(f"  MISSING {mod_name}")
            else:
                errors.append(f"{mod_name} failed to import: {e}")
                lines.append(f"  ERROR  {mod_name}: {e}")
        except Exception as e:
            errors.append(f"{mod_name} import failed: {type(e).__name__}: {e}")
            lines.append(f"  ERROR  {mod_name}: {type(e).__name__}: {e}")

    return lines


def _check_model_paths(errors: list[str], warnings: list[str]) -> list[str]:
    lines: list[str] = []
    for label, path_str in EXPECTED_MODEL_PATHS:
        p = Path(path_str)
        if not p.exists():
            errors.append(f"missing model path: {label} at {path_str}")
            lines.append(f"  MISSING {label:<24} {path_str}")
            continue
        if not p.is_dir():
            errors.append(f"{label} is not a directory: {path_str}")
            lines.append(f"  ERROR  {label:<24} {path_str} (not a directory)")
            continue

        # Quick size check — these directories should be large
        try:
            total_bytes = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            total_gb = total_bytes / 1024**3
        except Exception:
            total_gb = 0.0

        if total_gb < 0.1:
            warnings.append(f"{label} directory is suspiciously small ({total_gb:.2f} GB)")
            lines.append(f"  WARN   {label:<24} {path_str} ({total_gb:.2f} GB — too small?)")
        else:
            lines.append(f"  OK     {label:<24} {path_str} ({total_gb:.1f} GB)")

    return lines


# ──────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────

def run_preflight(verbose: bool = True) -> tuple[list[str], list[str]]:
    """Run all 8 preflight checks. Returns (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []

    def _emit(line: str) -> None:
        if verbose:
            print(line)

    _emit("")
    _emit("=" * 72)
    _emit(" ALLUVI — PREFLIGHT CHECK (local pipeline)")
    _emit("=" * 72)
    _emit("")

    _emit("[1/8] Checking .env keys + environment vars...")
    for line in _check_env_keys(errors):
        _emit(line)

    _emit("")
    _emit("[2/8] Checking asset files...")
    for line in _check_asset_files(errors):
        _emit(line)

    _emit("")
    _emit("[3/8] Checking brand & compliance files...")
    for line in _check_brand_files(errors, warnings):
        _emit(line)

    _emit("")
    _emit("[4/8] Checking scenarios & prompt templates...")
    for line in _check_scenarios_and_prompts(errors, warnings):
        _emit(line)

    _emit("")
    _emit("[5/8] Validating scenarios.yaml structure...")
    for line in _check_scenarios_yaml_structure(errors):
        _emit(line)

    _emit("")
    _emit("[6/8] Checking Python dependencies + CUDA + diffusers pipelines...")
    for line in _check_python_dependencies(errors, warnings):
        _emit(line)

    _emit("")
    _emit("[7/8] Checking project module imports...")
    for line in _check_project_imports(errors):
        _emit(line)

    _emit("")
    _emit("[8/8] Checking local model weights on disk...")
    for line in _check_model_paths(errors, warnings):
        _emit(line)

    _emit("")
    _emit("=" * 72)
    if not errors:
        _emit(" PREFLIGHT PASSED")
        _emit("=" * 72)
        if warnings:
            _emit("")
            _emit(f"{len(warnings)} non-blocking warning(s):")
            for w in warnings:
                _emit(f"  - {w}")
    else:
        _emit(f" PREFLIGHT FAILED — {len(errors)} error(s)")
        _emit("=" * 72)
        _emit("")
        _emit("Errors to fix before running:")
        _emit("")
        for e in errors:
            _emit(f"  X {e}")
        if warnings:
            _emit("")
            _emit(f"{len(warnings)} additional warning(s):")
            for w in warnings:
                _emit(f"  - {w}")
    _emit("")

    return errors, warnings


def main() -> int:
    errors, _warnings = run_preflight(verbose=True)
    if not errors:
        print("Ready to run:")
        print("  python orchestration/per_scenario/run.py --scenario <id>   # single scenario")
        print("  # (run_batch.py coming next)")
        print("")
        print("Estimated cost:  ~$0.28-$0.30 per scenario  (LLM only, GPU time separate)")
        print("Estimated wall:  ~3-5 min per scenario  (load+infer+unload per stage)")
        print("Peak VRAM:       ~48 GB (Qwen Stage 2)")
        print("")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())