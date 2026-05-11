"""
preflight.py — Pre-run validation for the Alluvi Final Image Generation pipeline.

Validates that every required file exists, every API key is set, every Python
dependency is installed, and every Python module is importable. Catches all
common setup errors BEFORE you spend any API credits on a real run.

Standalone:
    python preflight.py
    exits with code 0 on success, 1 on any error.

Auto-invoked:
    `run_batch.py` calls `run_preflight()` before its cost-confirmation prompt.
    If preflight fails, the batch never starts.

Adapted for the new repo from the production preflight.py with these changes:
  - Drops persona_face_only.jpg (new repo is PuLID-only, no Plan B)
  - Points to master_prompt_step2_qwen.md (the Qwen-tuned variant)
  - Checks httpx (not requests / PIL / openpyxl)
  - Checks new module structure: step_1_prompt_builder, step_2_prompt_builder,
    step_2_qwen_edit, db (not the old prompt_builder / step_2_nano_banana)
  - All paths anchored to REPO_ROOT (works from any CWD)
"""

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


# ──────────────────────────────────────────────────────────────────────────
# Individual checks — each returns (ok: bool, message_lines: list[str])
# ──────────────────────────────────────────────────────────────────────────

def _check_env_keys(errors: list[str]) -> list[str]:
    lines: list[str] = []
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
        for key_name in ("ANTHROPIC_API_KEY", "FAL_KEY"):
            if os.getenv(key_name):
                value = os.getenv(key_name)
                masked = value[:8] + "..." + value[-4:] if len(value) > 16 else "***"
                lines.append(f"  OK     {key_name} = {masked}")
            else:
                errors.append(f".env missing {key_name}")
                lines.append(f"  MISSING {key_name}")
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
    """Full archetype-aware validation of every scenario."""
    lines: list[str] = []
    scenarios_path = REPO_ROOT / "scenarios" / "scenarios.yaml"
    if not scenarios_path.exists():
        lines.append("  SKIP   scenarios.yaml missing (already reported in step 4)")
        return lines

    # Ensure REPO_ROOT is importable so we can use scenario_loader
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
        # ValueError from load_scenarios() contains ALL validation errors,
        # one per line — surface them so the user can fix in one pass.
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
        ("anthropic", "Claude Opus 4.7 SDK"),
        ("fal_client", "fal.ai endpoint client"),
        ("httpx", "HTTP downloads"),
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
                f"-- run `pip install -r requirements.txt`"
            )
            lines.append(f"  MISSING {module_name}  -- {role}")

    # sqlite3 is stdlib but verify it's available
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
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    src_modules = [
        "src.db",
        "src.scenario_loader",
        "src.step_1_prompt_builder",
        "src.step_1_pulid",
        "src.step_2_prompt_builder",
        "src.step_2_qwen_edit",
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


# ──────────────────────────────────────────────────────────────────────────
# Public entry point — used by both standalone CLI and run_batch.py
# ──────────────────────────────────────────────────────────────────────────

def run_preflight(verbose: bool = True) -> tuple[list[str], list[str]]:
    """
    Run all 7 preflight checks. Returns (errors, warnings).

    Args:
        verbose: if True, print check output to stdout as it runs.
                 If False, run silently (errors/warnings still returned).

    Returns:
        (errors, warnings) — empty errors list means preflight passed.
    """
    errors: list[str] = []
    warnings: list[str] = []

    def _emit(line: str) -> None:
        if verbose:
            print(line)

    _emit("")
    _emit("=" * 72)
    _emit(" ALLUVI — PREFLIGHT CHECK")
    _emit("=" * 72)
    _emit("")

    _emit("[1/7] Checking .env keys...")
    for line in _check_env_keys(errors):
        _emit(line)

    _emit("")
    _emit("[2/7] Checking asset files...")
    for line in _check_asset_files(errors):
        _emit(line)

    _emit("")
    _emit("[3/7] Checking brand & compliance files...")
    for line in _check_brand_files(errors, warnings):
        _emit(line)

    _emit("")
    _emit("[4/7] Checking scenarios & prompt templates...")
    for line in _check_scenarios_and_prompts(errors, warnings):
        _emit(line)

    _emit("")
    _emit("[5/7] Validating scenarios.yaml structure...")
    for line in _check_scenarios_yaml_structure(errors):
        _emit(line)

    _emit("")
    _emit("[6/7] Checking Python dependencies...")
    for line in _check_python_dependencies(errors):
        _emit(line)

    _emit("")
    _emit("[7/7] Checking project module imports...")
    for line in _check_project_imports(errors):
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


# ──────────────────────────────────────────────────────────────────────────
# Standalone CLI entry point
# ──────────────────────────────────────────────────────────────────────────

def main() -> int:
    errors, _warnings = run_preflight(verbose=True)
    if not errors:
        print("Ready to run:")
        print("  python run.py --scenario <id>            # single scenario test")
        print("  python run_batch.py --pilot              # 5 pilot scenarios")
        print("  python run_batch.py                      # all 30 scenarios")
        print("")
        print("Estimated cost:  ~$0.36 per scenario  (~$10.80 for full 30)")
        print("Estimated time:  ~70-80s per scenario (~35-40 min for full 30, sequential)")
        print("")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())