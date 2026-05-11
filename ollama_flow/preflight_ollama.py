"""
ollama_flow/preflight_ollama.py — pre-run validation for the Ollama flow.

Mirrors the production preflight.py's 7-step pattern but tuned for Ollama mode:
  - Checks ollama_flow/.env (only FAL_KEY required, no ANTHROPIC_API_KEY needed)
  - Verifies Ollama server is reachable and the configured model is pulled
  - Shares parent's assets/brand/prompts/scenarios (no duplication)
  - Validates scenarios.yaml via parent's scenario_loader
  - Checks Ollama-flow modules import cleanly

Standalone:
    python preflight_ollama.py
    exits 0 on success, 1 on any error.

Auto-invoked:
    run_batch_ollama.py calls run_preflight() before its cost-confirmation prompt.
"""

import os
import sys
from pathlib import Path


# This file lives at  Final_Image_generation/ollama_flow/preflight_ollama.py
# Parent repo root is Final_Image_generation/    (one level up from this file)
OLLAMA_FLOW_ROOT = Path(__file__).resolve().parent
PARENT_REPO_ROOT = OLLAMA_FLOW_ROOT.parent


# Make both folders importable. Parent goes in LAST so it ends up at
# sys.path[0] and wins name lookups. (sys.path.insert(0, x) prepends —
# last insert becomes first entry.)
for p in (str(OLLAMA_FLOW_ROOT), str(PARENT_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


# ──────────────────────────────────────────────────────────────────────────
# Individual checks — each returns a list of output lines
# ──────────────────────────────────────────────────────────────────────────

def _check_env_keys(errors: list[str]) -> list[str]:
    """
    Ollama mode needs FAL_KEY only. ANTHROPIC_API_KEY is NOT required here.
    """
    lines: list[str] = []
    try:
        from dotenv import load_dotenv

        load_dotenv(OLLAMA_FLOW_ROOT / ".env")

        # FAL_KEY required
        if os.getenv("FAL_KEY"):
            value = os.getenv("FAL_KEY")
            masked = value[:8] + "..." + value[-4:] if len(value) > 16 else "***"
            lines.append(f"  OK     FAL_KEY = {masked}")
        else:
            errors.append("ollama_flow/.env missing FAL_KEY")
            lines.append("  MISSING FAL_KEY (required for fal Stage 1 + Stage 2 calls)")

        # Ollama overrides (informational)
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        lines.append(f"  INFO   OLLAMA_HOST  = {host}")
        lines.append(f"  INFO   OLLAMA_MODEL = {model}")

    except ImportError:
        errors.append("python-dotenv not installed: run `pip install -r ../requirements.txt`")
        lines.append("  ERROR  python-dotenv not installed")
    return lines


def _check_ollama_reachability(errors: list[str]) -> list[str]:
    """Verify Ollama server responds and the configured model is pulled."""
    lines: list[str] = []
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

    try:
        import httpx
    except ImportError:
        errors.append("httpx not installed — run `pip install -r ../requirements.txt`")
        lines.append("  ERROR  httpx not installed")
        return lines

    # Ping /api/tags to confirm server is up AND get the model list
    try:
        resp = httpx.get(host.rstrip("/") + "/api/tags", timeout=5)
    except httpx.ConnectError as e:
        errors.append(
            f"Cannot reach Ollama at {host}. "
            f"Is `ollama serve` running? Underlying error: {e}"
        )
        lines.append(f"  ERROR  Ollama unreachable at {host}")
        return lines
    except Exception as e:
        errors.append(f"Ollama check failed: {type(e).__name__}: {e}")
        lines.append(f"  ERROR  {type(e).__name__}: {e}")
        return lines

    if resp.status_code != 200:
        errors.append(f"Ollama returned HTTP {resp.status_code} at {host}/api/tags")
        lines.append(f"  ERROR  Ollama HTTP {resp.status_code}")
        return lines

    try:
        data = resp.json()
        tags = data.get("models", [])
    except Exception as e:
        errors.append(f"Could not parse Ollama /api/tags response: {e}")
        lines.append(f"  ERROR  Could not parse Ollama response")
        return lines

    available = [t.get("name", "") for t in tags]
    if any(model in name for name in available):
        # Find the matching tag for size info
        matching = next((t for t in tags if model in t.get("name", "")), None)
        size_str = ""
        if matching:
            size_bytes = matching.get("size", 0)
            if size_bytes:
                size_gb = size_bytes / (1024 ** 3)
                size_str = f" ({size_gb:.1f} GB)"
        lines.append(f"  OK     Ollama reachable, model '{model}' is pulled{size_str}")
    else:
        errors.append(
            f"Ollama reachable at {host} but model '{model}' is not pulled. "
            f"Run `ollama pull {model}` first."
        )
        sample = available[:5] if available else ["(none)"]
        lines.append(f"  MISSING Ollama model '{model}' (available: {sample})")

    return lines


def _check_shared_asset_files(errors: list[str]) -> list[str]:
    """Asset files live in the PARENT repo — we share them."""
    lines: list[str] = []
    asset_files = [
        ("assets/persona.jpg", "Step 1 PuLID reference (full body)"),
        ("assets/persona.yaml", "Persona identity descriptor"),
        ("assets/product.jpg", "Step 2 product reference"),
        ("assets/product.yaml", "Product packaging descriptor"),
    ]
    for fpath, role in asset_files:
        p = PARENT_REPO_ROOT / fpath
        if p.exists():
            size_kb = p.stat().st_size / 1024
            lines.append(f"  OK     ../{fpath:<32} ({size_kb:>6.1f} KB)  -- {role}")
        else:
            errors.append(f"missing asset file: ../{fpath} ({role})")
            lines.append(f"  MISSING ../{fpath}  -- {role}")
    return lines


def _check_shared_brand_files(errors: list[str], warnings: list[str]) -> list[str]:
    """Brand files live in the PARENT repo — we share them."""
    lines: list[str] = []
    brand_files = [
        "brand/brand.yaml",
        "brand/do_dont.md",
    ]
    for fpath in brand_files:
        p = PARENT_REPO_ROOT / fpath
        if p.exists() and p.stat().st_size > 100:
            lines.append(f"  OK     ../{fpath} ({p.stat().st_size:,} bytes)")
        elif p.exists():
            warnings.append(f"../{fpath} exists but is suspiciously small")
            lines.append(f"  WARN   ../{fpath} (only {p.stat().st_size} bytes)")
        else:
            errors.append(f"missing brand file: ../{fpath}")
            lines.append(f"  MISSING ../{fpath}")
    return lines


def _check_shared_scenarios_and_prompts(errors: list[str], warnings: list[str]) -> list[str]:
    """Scenarios + prompts live in the PARENT repo — we share them."""
    lines: list[str] = []
    required_files = [
        ("scenarios/scenarios.yaml", "30 hand-curated scenarios"),
        ("prompts/master_prompt_step1.md", "Step 1 PuLID system prompt"),
        ("prompts/master_prompt_step2_qwen.md", "Step 2 Qwen-tuned system prompt"),
    ]
    for fpath, role in required_files:
        p = PARENT_REPO_ROOT / fpath
        if p.exists() and p.stat().st_size > 1000:
            lines.append(f"  OK     ../{fpath:<45} ({p.stat().st_size:>6,} bytes)  -- {role}")
        elif p.exists():
            warnings.append(f"../{fpath} exists but is small ({p.stat().st_size} bytes)")
            lines.append(f"  WARN   ../{fpath} (only {p.stat().st_size} bytes)")
        else:
            errors.append(f"missing required file: ../{fpath}")
            lines.append(f"  MISSING ../{fpath}")
    return lines


def _check_scenarios_yaml_structure(errors: list[str]) -> list[str]:
    """Full archetype-aware validation via the parent's scenario_loader."""
    lines: list[str] = []
    scenarios_path = PARENT_REPO_ROOT / "scenarios" / "scenarios.yaml"
    if not scenarios_path.exists():
        lines.append("  SKIP   scenarios.yaml missing (already reported in step 4)")
        return lines

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
        errors.append(f"could not import scenario_loader from parent: {e}")
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


def _check_python_dependencies(errors: list[str]) -> list[str]:
    lines: list[str] = []
    deps = [
        ("fal_client", "fal.ai endpoint client"),
        ("httpx", "HTTP — used by both fal downloads and Ollama calls"),
        ("dotenv", "via python-dotenv"),
        ("yaml", "via PyYAML"),
    ]
    for module_name, role in deps:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "?")
            lines.append(f"  OK     {module_name:<13} {version:<12}  -- {role}")
        except ImportError:
            errors.append(
                f"missing dependency: {module_name} "
                f"-- run `pip install -r ../requirements.txt`"
            )
            lines.append(f"  MISSING {module_name}  -- {role}")

    try:
        import sqlite3
        lines.append(
            f"  OK     sqlite3       {sqlite3.sqlite_version:<12}  -- stdlib (database)"
        )
    except ImportError:
        errors.append("sqlite3 not available (Python stdlib problem)")
        lines.append("  ERROR  sqlite3 not available")

    return lines


def _check_project_imports(errors: list[str]) -> list[str]:
    lines: list[str] = []

    # Parent modules we reuse
    parent_modules = [
        "src.db",
        "src.scenario_loader",
        "src.step_1_pulid",
        "src.step_2_qwen_edit",
        "src.trace_html",
        "src.overview_html",
    ]
    for mod_name in parent_modules:
        try:
            __import__(mod_name)
            lines.append(f"  OK     parent: {mod_name}")
        except ModuleNotFoundError as e:
            errors.append(f"missing parent module: {mod_name.replace('.', '/')}.py")
            lines.append(f"  MISSING parent: {mod_name}")
        except Exception as e:
            errors.append(f"{mod_name} import failed: {type(e).__name__}: {e}")
            lines.append(f"  ERROR  parent: {mod_name}: {type(e).__name__}: {e}")

    # Ollama-flow modules — plain imports work now that the package is named
    # `ollama_src` (no collision with parent's `src`).
    ollama_modules = [
        "ollama_src.ollama_client",
        "ollama_src.step_1_prompt_builder_ollama",
        "ollama_src.step_2_prompt_builder_ollama",
    ]
    for mod_name in ollama_modules:
        try:
            __import__(mod_name)
            lines.append(f"  OK     ollama: {mod_name}")
        except ModuleNotFoundError as e:
            errors.append(f"missing ollama module: {mod_name.replace('.', '/')}.py")
            lines.append(f"  MISSING ollama: {mod_name}")
        except Exception as e:
            errors.append(f"{mod_name} import failed: {type(e).__name__}: {e}")
            lines.append(f"  ERROR  ollama: {mod_name}: {type(e).__name__}: {e}")

    return lines


# ──────────────────────────────────────────────────────────────────────────
# Public entry point — used by standalone CLI + run_batch_ollama
# ──────────────────────────────────────────────────────────────────────────

def run_preflight(verbose: bool = True) -> tuple[list[str], list[str]]:
    """
    Run all 7 preflight checks. Returns (errors, warnings).
    Empty errors list = preflight passed.
    """
    errors: list[str] = []
    warnings: list[str] = []

    def _emit(line: str) -> None:
        if verbose:
            print(line)

    _emit("")
    _emit("=" * 72)
    _emit(" ALLUVI — OLLAMA FLOW PREFLIGHT CHECK")
    _emit("=" * 72)
    _emit("")

    _emit("[1/7] Checking .env keys (Ollama mode: FAL_KEY only)...")
    for line in _check_env_keys(errors):
        _emit(line)

    _emit("")
    _emit("[2/7] Checking Ollama server reachability + model availability...")
    for line in _check_ollama_reachability(errors):
        _emit(line)

    _emit("")
    _emit("[3/7] Checking shared asset files (from parent repo)...")
    for line in _check_shared_asset_files(errors):
        _emit(line)

    _emit("")
    _emit("[4/7] Checking shared brand & scenarios (from parent repo)...")
    for line in _check_shared_brand_files(errors, warnings):
        _emit(line)
    for line in _check_shared_scenarios_and_prompts(errors, warnings):
        _emit(line)

    _emit("")
    _emit("[5/7] Validating scenarios.yaml structure (via parent's scenario_loader)...")
    for line in _check_scenarios_yaml_structure(errors):
        _emit(line)

    _emit("")
    _emit("[6/7] Checking Python dependencies...")
    for line in _check_python_dependencies(errors):
        _emit(line)

    _emit("")
    _emit("[7/7] Checking project module imports (parent + ollama-flow)...")
    for line in _check_project_imports(errors):
        _emit(line)

    _emit("")
    _emit("=" * 72)
    if not errors:
        _emit(" OLLAMA-FLOW PREFLIGHT PASSED")
        _emit("=" * 72)
        if warnings:
            _emit("")
            _emit(f"{len(warnings)} non-blocking warning(s):")
            for w in warnings:
                _emit(f"  - {w}")
    else:
        _emit(f" OLLAMA-FLOW PREFLIGHT FAILED — {len(errors)} error(s)")
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
        print("Ready to run (Ollama mode):")
        print("  python run_ollama.py --scenario <id>            # single scenario test")
        print("  python run_batch_ollama.py --pilot              # 5 pilot scenarios")
        print("  python run_batch_ollama.py                      # all 30 scenarios")
        print("")
        print("Estimated cost:  ~$0.08 per scenario  (~$2.40 for full 30)")
        print("                 — fal API only, LLM is free via local Ollama")
        print("Estimated time:  ~80-120s per scenario (Ollama latency adds vs. Claude)")
        print("")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())