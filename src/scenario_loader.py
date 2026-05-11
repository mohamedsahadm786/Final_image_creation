"""
src/scenario_loader.py — Read and validate scenarios.yaml.

Production-extracted verbatim from src/scenario_loader.py with one path
adaptation: SCENARIOS_PATH anchored to REPO_ROOT instead of CWD-relative.

Provides:
  - load_scenarios()                : returns the full list of scenarios
  - load_scenario(scenario_id)      : returns one scenario by id
  - filter_pilot_scenarios()        : returns the easy/diagnostic subset
  - validate_scenario(record)       : raises ValueError if a scenario is malformed

Validation is ARCHETYPE-AWARE. Different archetype shapes have different
required field sets:

  Persona-present archetypes (held_*, placed_on_surface with persona):
      require: outfit, pose, hand_assignment, grip_or_placement,
               lighting, mood, palette, framing, camera_height

  Persona-absent archetypes (flat_lay, object_in_lineup, placed_on_surface
  without persona — e.g. scenario_07):
      require: scene, grip_or_placement, lighting, mood, palette,
               framing, camera_height
      do NOT require: outfit, pose, hand_assignment

  `placed_on_surface` is a special case — some scenarios have a partial
  persona in frame (e.g. scenario_19 with shoulder/ear visible) and some
  don't (e.g. scenario_07, pilates floor). The validator treats `outfit`
  presence as the signal: if outfit is set and non-empty, persona-present
  fields are required; if outfit is null/missing, only persona-absent
  fields are required.
"""

from pathlib import Path
from typing import Any
import yaml

# Repo-root-anchored path (changed from CWD-relative for the new repo)
REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_PATH = REPO_ROOT / "scenarios" / "scenarios.yaml"


# ──────────────────────────────────────────────────────────────────────────
# REQUIRED FIELDS BY ARCHETYPE
# ──────────────────────────────────────────────────────────────────────────

# Fields every scenario MUST have, regardless of archetype:
ALWAYS_REQUIRED_FIELDS = [
    "id",
    "category",
    "archetype",
    "difficulty",
    "scene",
    "grip_or_placement",   # how/where the product is held or placed
    "lighting",
    "mood",
    "palette",
    "framing",
    "camera_height",
]

# Fields required ONLY when a persona is present in the frame:
PERSONA_PRESENT_FIELDS = [
    "outfit",
    "pose",
    "hand_assignment",
]

# Archetypes where the persona is ALWAYS in frame:
PERSONA_ALWAYS_PRESENT = {
    "held_product_high",
    "held_product_low",
    "held_with_phone",
}

# Archetypes where the persona is NEVER in frame:
PERSONA_NEVER_PRESENT = {
    "flat_lay",
    "object_in_lineup",
}

# Archetypes where persona presence varies per-scenario.
# For these we infer presence from whether `outfit` is set.
PERSONA_VARIABLE = {
    "placed_on_surface",
}

VALID_ARCHETYPES = (
    PERSONA_ALWAYS_PRESENT | PERSONA_NEVER_PRESENT | PERSONA_VARIABLE
)

VALID_DIFFICULTIES = {"easy", "medium", "hard"}


# ──────────────────────────────────────────────────────────────────────────
# LOADERS
# ──────────────────────────────────────────────────────────────────────────

def load_scenarios() -> list[dict[str, Any]]:
    """Load and validate all scenarios from scenarios.yaml."""
    if not SCENARIOS_PATH.exists():
        raise FileNotFoundError(f"scenarios.yaml not found at {SCENARIOS_PATH}")

    with SCENARIOS_PATH.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "scenarios" not in data:
        raise ValueError("scenarios.yaml must have a top-level `scenarios:` list")

    scenarios = data["scenarios"]
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("scenarios.yaml `scenarios:` is empty or not a list")

    # Validate every scenario (collect ALL errors, not just the first)
    errors: list[str] = []
    for sc in scenarios:
        try:
            validate_scenario(sc)
        except ValueError as e:
            errors.append(str(e))

    if errors:
        msg = f"scenarios.yaml has {len(errors)} validation error(s):\n"
        msg += "\n".join(f"  - {e}" for e in errors)
        raise ValueError(msg)

    # Check for duplicate IDs
    ids = [s["id"] for s in scenarios]
    duplicates = [i for i in ids if ids.count(i) > 1]
    if duplicates:
        raise ValueError(f"scenarios.yaml has duplicate ids: {sorted(set(duplicates))}")

    return scenarios


def load_scenario(scenario_id: str) -> dict[str, Any]:
    """Load a single scenario by its id."""
    for sc in load_scenarios():
        if sc["id"] == scenario_id:
            return sc
    raise ValueError(f"scenario id not found: {scenario_id}")


def filter_pilot_scenarios(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Return the pilot subset:
      - All 3 hero flat-lays (easy validation)
      - 2 easy non-hero scenarios for cross-category validation

    Total: 5 scenarios. Lowest-risk way to validate the full pipeline before
    spending on the full 30-image run.
    """
    hero_flat_lays = [s for s in scenarios if s["category"] == "hero_flat_lay"]
    easy_others = [
        s
        for s in scenarios
        if s["difficulty"] == "easy" and s["category"] != "hero_flat_lay"
    ][:2]
    return hero_flat_lays + easy_others


# ──────────────────────────────────────────────────────────────────────────
# VALIDATION
# ──────────────────────────────────────────────────────────────────────────

def _persona_is_present(sc: dict[str, Any]) -> bool:
    """
    Determine whether this scenario has a persona in frame.

    Rules:
      - Archetypes in PERSONA_ALWAYS_PRESENT  → always True
      - Archetypes in PERSONA_NEVER_PRESENT   → always False
      - Archetypes in PERSONA_VARIABLE        → True iff `outfit` is non-empty
    """
    archetype = sc.get("archetype")
    if archetype in PERSONA_ALWAYS_PRESENT:
        return True
    if archetype in PERSONA_NEVER_PRESENT:
        return False
    # PERSONA_VARIABLE: infer from outfit presence
    outfit = sc.get("outfit")
    return bool(outfit) and outfit not in ("", "~", None)


def validate_scenario(sc: dict[str, Any]) -> None:
    """Raise ValueError if the scenario is missing required fields or has bad values."""
    if not isinstance(sc, dict):
        raise ValueError(f"scenario must be a dict, got {type(sc).__name__}")

    sc_id = sc.get("id", "<unknown id>")

    # 1. Always-required fields
    for field in ALWAYS_REQUIRED_FIELDS:
        if field not in sc:
            raise ValueError(f"scenario {sc_id} missing required field: {field}")

    # 2. Validate archetype value
    if sc["archetype"] not in VALID_ARCHETYPES:
        raise ValueError(
            f"scenario {sc_id} has invalid archetype '{sc['archetype']}'. "
            f"Valid: {sorted(VALID_ARCHETYPES)}"
        )

    # 3. Validate difficulty value
    if sc["difficulty"] not in VALID_DIFFICULTIES:
        raise ValueError(
            f"scenario {sc_id} has invalid difficulty '{sc['difficulty']}'. "
            f"Valid: {sorted(VALID_DIFFICULTIES)}"
        )

    # 4. Persona-present field check (archetype-aware)
    if _persona_is_present(sc):
        for field in PERSONA_PRESENT_FIELDS:
            if not sc.get(field):
                raise ValueError(
                    f"scenario {sc_id} missing `{field}` "
                    f"(required for persona-present archetype={sc['archetype']})"
                )
        # hand_assignment must have the 4 sub-keys when persona is present
        ha = sc["hand_assignment"]
        if not isinstance(ha, dict):
            raise ValueError(
                f"scenario {sc_id} hand_assignment must be a dict, got {type(ha).__name__}"
            )
        for sub in ("phone_hand", "product_hand", "other_hand", "other_hand_does"):
            if sub not in ha:
                raise ValueError(
                    f"scenario {sc_id} hand_assignment missing key: {sub}"
                )
    # else: persona-absent — no outfit/pose/hand_assignment required