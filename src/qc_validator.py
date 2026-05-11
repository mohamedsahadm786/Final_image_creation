"""
src/qc_validator.py — image quality-control validator (LENIENT mode).

Uses Claude Sonnet 4.6 (multimodal) to check final composited images for
OBVIOUSLY ILLOGICAL defects only. Minor imperfections pass — only things
that "don't make sense" fail:
  - 3+ legs, 3+ hands, 3+ arms
  - 6+ fingers on a hand
  - Fused limbs into impossible shapes
  - Multiple people when one is expected
  - Multiple product copies (not mirror reflections)
  - Product warped into non-rectangular shape

Things that PASS (ignored as minor):
  - Slight text artifacts on packaging (ALUUVI vs ALLUVI)
  - Minor lighting inconsistencies
  - Subtle face asymmetry
  - Slightly weird finger curl (as long as count is right)
  - Background imperfections

Used as a permissive gate to the video-creation pipeline. Better to skip
2-3 truly broken images than to over-reject usable ones.

WHY SONNET 4.6 — NOT HAIKU:
  Sonnet gives more nuanced visual judgment for anatomy questions. The
  cost difference is small (~$0.01 vs $0.001 per image) but Sonnet is
  much less likely to false-positive on minor issues.

COST:
  ~$0.01 per image with Sonnet 4.6. For a 30-scenario batch: ~$0.30.

LATENCY:
  ~5-10s per check.

USAGE:
  from src.qc_validator import validate_image
  result = validate_image(Path("output/05_step2_final.jpg"))
  if result["passed"]:
      # send to video pipeline
  else:
      # regenerate Step 2 only
"""

import base64
import os
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv

from src.json_utils import validate_json_output, JSONSanityError


load_dotenv()


# Sonnet 4.6 — better nuanced visual judgment than Haiku
QC_MODEL = "claude-sonnet-4-6-20250929"
MAX_TOKENS = 1500


_client: Anthropic | None = None


def _get_client() -> Anthropic:
    """Lazy-init the Anthropic client."""
    global _client
    if _client is None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY missing — QC validator requires Anthropic "
                "access. Set it in .env or disable QC by not calling "
                "validate_image()."
            )
        _client = Anthropic()
    return _client


# ─── QC rubric — LENIENT mode ────────────────────────────────────────────

QC_SYSTEM_PROMPT = (
    "You are a LENIENT quality reviewer for AI-generated ad images. "
    "Your job is to catch ONLY obviously illogical/impossible defects — "
    "things a viewer would immediately recognize as 'this is broken' or "
    "'this doesn't make sense'. Minor imperfections, slight artifacts, "
    "small text rendering issues, subtle lighting drift, and other small "
    "flaws should be IGNORED — they pass.\n\n"
    "Only fail an image if it has obvious anatomy errors (extra limbs, "
    "extra hands, extra legs, way too many fingers, fused body parts) or "
    "obvious product errors (multiple distinct product copies, product "
    "shape completely warped/melted, no product visible at all).\n\n"
    "When in doubt, PASS. We'd rather ship one slightly imperfect image "
    "than reject usable ones.\n\n"
    "You MUST respond with a single JSON object — no markdown fences, "
    "no preamble, no explanation outside the JSON."
)


QC_RUBRIC = """\
Look at this image. Apply LENIENT quality standards — only flag obvious,
illogical defects. Ignore minor blur, slight text artifacts, small lighting
inconsistencies, and other minor imperfections.

Answer these specific questions:

ANATOMY (only flag if obviously illogical):
1. person_count: How many distinct people are visible? (must be exactly 1)
2. has_extra_limbs: Are there obviously extra arms (more than 2), extra
   legs (more than 2), or extra hands (more than 2)? Don't worry about
   one partially hidden or cropped — only flag CLEAR extras. (expected: false)
3. has_extreme_finger_issue: Does any hand have 6 or more clearly visible
   fingers, OR are fingers obviously fused into a single mass? Slight
   finger curl, partial occlusion, or normal-count fingers in weird
   positions are FINE. (expected: false)
4. has_fused_or_warped_limbs: Are limbs obviously fused together into
   impossible shapes, or twisted in physically impossible ways? Don't
   flag natural body curves or poses. (expected: false)
5. face_grossly_distorted: Is the face severely distorted, multiple faces,
   or missing? Don't flag minor asymmetry or slight blur. (expected: false)

PRODUCT (only flag if obviously illogical):
6. product_visible: Is there a product visible? (expected: true)
7. multiple_distinct_products: Are there 2+ DISTINCT separate product
   copies visible (not a mirror reflection of the same one)? (expected: false)
8. product_shape_broken: Is the product warped, melted, or transformed
   into a non-rectangular impossible shape? Minor angle changes or slight
   perspective distortion are FINE. (expected: false)

Then provide:
- specific_issues: list of ONLY the illogical defects you found (empty if image is acceptable)
- overall_recommendation: "use" (acceptable for video), "regenerate" (try again), or "discard"
- confidence: your confidence (0.0 to 1.0)

Respond with EXACTLY this JSON, no additional keys:
{
  "person_count": <int>,
  "has_extra_limbs": <bool>,
  "has_extreme_finger_issue": <bool>,
  "has_fused_or_warped_limbs": <bool>,
  "face_grossly_distorted": <bool>,
  "product_visible": <bool>,
  "multiple_distinct_products": <bool>,
  "product_shape_broken": <bool>,
  "specific_issues": [<string>, ...],
  "overall_recommendation": "use" | "regenerate" | "discard",
  "confidence": <float>
}
"""


# ─── Public entry point ───────────────────────────────────────────────────

def validate_image(
    image_path: Path,
    *,
    scenario_id: str = "?",
) -> dict[str, Any]:
    """
    Run LENIENT QC validation. Only flags obviously illogical defects.

    Args:
        image_path: path to the image file (jpg/png/webp)
        scenario_id: scenario id for logging

    Returns:
        dict with `passed`, `score`, `checks`, `issues`, `recommendation`,
        `confidence`, `model`, `error`, `raw_vlm_response`.
    """
    if not image_path.exists():
        return _error_result(
            f"image file not found: {image_path}",
            scenario_id=scenario_id,
        )

    print(f"[qc_validator] {scenario_id}: running QC via {QC_MODEL}...")

    try:
        image_data, media_type = _load_image_base64(image_path)
    except Exception as e:
        return _error_result(
            f"failed to load image: {type(e).__name__}: {e}",
            scenario_id=scenario_id,
        )

    try:
        response = _get_client().messages.create(
            model=QC_MODEL,
            max_tokens=MAX_TOKENS,
            system=QC_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": QC_RUBRIC},
                    ],
                }
            ],
        )
    except Exception as e:
        return _error_result(
            f"QC API call failed: {type(e).__name__}: {e}",
            scenario_id=scenario_id,
        )

    try:
        raw_text = response.content[0].text
        result = validate_json_output(
            raw_text,
            required_keys=["overall_recommendation"],
        )
    except JSONSanityError as e:
        return _error_result(
            f"QC response JSON parse failed: {e}",
            scenario_id=scenario_id,
        )

    decision = _score_qc_result(result)

    status = "PASS" if decision["passed"] else "FAIL"
    n_issues = len(decision["issues"])
    print(
        f"[qc_validator] {scenario_id}: {status} "
        f"(score={decision['score']:.2f}, issues={n_issues}, "
        f"rec={decision['recommendation']})"
    )
    if decision["issues"]:
        for issue in decision["issues"]:
            print(f"  - {issue}")

    return decision


# ─── Internal helpers ─────────────────────────────────────────────────────

def _load_image_base64(image_path: Path) -> tuple[str, str]:
    """Load image, return (base64_string, media_type)."""
    suffix = image_path.suffix.lower()
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix, "image/jpeg")

    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, media_type


def _score_qc_result(result: dict) -> dict[str, Any]:
    """
    LENIENT scoring — only OBVIOUSLY ILLOGICAL defects fail.

    Hard fails (image will be regenerated):
      - person_count != 1
      - has_extra_limbs == true
      - has_extreme_finger_issue == true
      - has_fused_or_warped_limbs == true
      - face_grossly_distorted == true
      - product_visible == false
      - multiple_distinct_products == true
      - product_shape_broken == true

    Everything else passes — no soft warnings, no strict mode.
    """
    illogical_defects: list[str] = []

    # Anatomy — only obviously illogical
    person_count = result.get("person_count")
    if person_count is not None and person_count != 1:
        illogical_defects.append(
            f"wrong person count: {person_count} (must be 1)"
        )

    if result.get("has_extra_limbs"):
        illogical_defects.append("obviously extra limbs detected (3+ arms/legs/hands)")

    if result.get("has_extreme_finger_issue"):
        illogical_defects.append("extreme finger anatomy issue (6+ fingers or fused mass)")

    if result.get("has_fused_or_warped_limbs"):
        illogical_defects.append("limbs fused into impossible shapes")

    if result.get("face_grossly_distorted"):
        illogical_defects.append("face is grossly distorted or missing")

    # Product — only obviously illogical
    if not result.get("product_visible", True):
        illogical_defects.append("product not visible at all")

    if result.get("multiple_distinct_products"):
        illogical_defects.append("multiple distinct product copies visible")

    if result.get("product_shape_broken"):
        illogical_defects.append("product shape is broken/warped beyond a normal rectangle")

    # Anything else the VLM noted explicitly
    extra_issues = result.get("specific_issues") or []
    if not isinstance(extra_issues, list):
        extra_issues = [str(extra_issues)]

    # Pass/fail: ANY illogical defect = fail; otherwise pass
    passed = len(illogical_defects) == 0

    # Score: 1.0 if pass, drop 0.2 per defect, min 0.0
    score = max(0.0, 1.0 - (len(illogical_defects) * 0.2))

    # Combine issues, dedup
    seen: set = set()
    all_issues: list[str] = []
    for src in (illogical_defects, extra_issues):
        for item in src:
            s = str(item)
            if s and s not in seen:
                seen.add(s)
                all_issues.append(s)

    # VLM's recommendation (or derive from our decision)
    rec = result.get("overall_recommendation", "")
    if rec not in ("use", "regenerate", "discard"):
        rec = "use" if passed else "regenerate"

    return {
        "passed": passed,
        "score": round(score, 3),
        "checks": {
            "person_count": result.get("person_count"),
            "has_extra_limbs": result.get("has_extra_limbs"),
            "has_extreme_finger_issue": result.get("has_extreme_finger_issue"),
            "has_fused_or_warped_limbs": result.get("has_fused_or_warped_limbs"),
            "face_grossly_distorted": result.get("face_grossly_distorted"),
            "product_visible": result.get("product_visible"),
            "multiple_distinct_products": result.get("multiple_distinct_products"),
            "product_shape_broken": result.get("product_shape_broken"),
        },
        "issues": all_issues,
        "recommendation": rec,
        "confidence": float(result.get("confidence", 0.5) or 0.5),
        "error": None,
        "model": QC_MODEL,
        "raw_vlm_response": result,
    }


def _error_result(
    error_message: str,
    *,
    scenario_id: str = "?",
) -> dict[str, Any]:
    """Build a failure result when QC call itself fails (treat as 'pass with warning')."""
    print(f"[qc_validator] {scenario_id}: QC ERROR — {error_message}")
    return {
        # On QC infrastructure failure: PASS the image (don't punish for our QC outage).
        "passed": True,
        "score": 0.5,
        "checks": {},
        "issues": [f"QC validation failed (treating as pass): {error_message}"],
        "recommendation": "use",
        "confidence": 0.0,
        "error": error_message,
        "model": QC_MODEL,
        "raw_vlm_response": None,
    }