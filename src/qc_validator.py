"""
src/qc_validator.py — image quality-control validator (BALANCED mode).

Uses Claude Sonnet 4.5 (multimodal) to gate AI-generated product-ad images
before they go to the video pipeline.

KEY TECHNIQUE — DESCRIBE-THEN-JUDGE FOR LIMBS:
  A plain yes/no "is the hand malformed?" question is unreliable — the model
  tends to answer "false" on borderline cases. So for hands / palms / arms /
  legs the rubric FORCES the model to first WRITE OUT a description of each
  one (how many palms on each wrist, how many hands on each arm, any extra
  mass) and ONLY THEN set the boolean flags. Describing the defect out loud
  makes a wrong "clean" verdict far less likely. This applies to limbs only —
  finger-count and product checks are unchanged and stay lenient.

FAILS (image is regenerated):
  ANATOMY
    - more than 1 person
    - more than 2 arms / 2 hands / 2 legs (clear extras)
    - a malformed hand: a doubled palm, a second palm fused to the same
      wrist, an extra hand-mass growing out of one hand, or a hand/arm
      attached at an impossible place
    - a BLATANT extra-finger blunder: 7+ clearly visible fingers on one
      hand, or fingers fused into a single mass
    - limbs fused / twisted into physically impossible shapes
    - face grossly distorted or missing
  PLACEMENT (logical, not perfect)
    - hand/grip clearly illogical — product floating, impossible grip
    - leg/body pose physically absurd
  PRODUCT
    - no product visible at all
    - 2+ distinct product copies
    - box warped/melted into a non-rectangular impossible shape
    - brand name "ALLUVI" less than ~90% legible  (THE strict check)
    - product name "TIRZEPATIDE" less than ~70% resemblance (looser)
    - box structure/colour theme less than ~80%

ALWAYS PASSES (ignored — minor):
  - finger COUNT when fingers are hidden/masked by the product (NEVER fail
    for "too few" fingers)
  - tiny finger-curl weirdness, partial occlusion
  - minor lighting / colour drift, soft focus, slight blur
  - background imperfections
  - small text artifacts above the legibility thresholds
  - slightly imperfect but plausible poses

NOTE: product-orientation checking is intentionally NOT done here yet — it
will be added after the product LoRA, together with Option B (real product
reference image passed to QC).

COST:   ~$0.01 per image (Sonnet 4.5).
LATENCY: ~5-10s per check.

USAGE:
  from src.qc_validator import validate_image
  result = validate_image(Path("output/05_step2_final.jpg"))
  if result["passed"]:
      ...                          # send to video pipeline
  else:
      defects = result["issues"]   # short imperative phrases for retry
"""

import base64
import os
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv

from src.json_utils import validate_json_output, JSONSanityError


load_dotenv()


QC_MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 2000


_client: Anthropic | None = None


def _get_client() -> Anthropic:
    """Lazy-init the Anthropic client."""
    global _client
    if _client is None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY missing — QC validator requires Anthropic "
                "access. Set it in .env or disable QC."
            )
        _client = Anthropic()
    return _client


# ─── QC system prompt ────────────────────────────────────────────────────

QC_SYSTEM_PROMPT = (
    "You are a BALANCED quality reviewer for AI-generated product-ad images. "
    "The image will become the first frame of a short video ad for a "
    "weight-management product called Alluvi.\n\n"
    "Your job: catch BLUNDERS, not minor imperfections. The image must look "
    "LOGICAL and believable. Slight blur, minor lighting drift, tiny "
    "imperfections, and small artifacts are FINE and must PASS.\n\n"
    "You are especially careful and STRICT about HANDS, PALMS, ARMS and LEGS. "
    "AI image models frequently produce a doubled palm, a second palm on one "
    "wrist, an extra hand-mass, or a limb attached at an impossible place. "
    "These are serious defects. To catch them you must FIRST describe each "
    "hand, arm and leg in plain words, THEN judge — never judge before "
    "describing.\n\n"
    "IMPORTANT about fingers: the product often hides part of the hand. "
    "Fingers behind or wrapped around the box are naturally not visible. "
    "NEVER fail an image for 'too few' fingers — that is expected. Only flag "
    "fingers for a BLATANT blunder: 7+ clearly visible fingers on one hand, "
    "or fingers fused into a single mass.\n\n"
    "You MUST respond with a single JSON object — no markdown fences, no "
    "preamble, no text outside the JSON."
)


QC_RUBRIC = """\
Review this image. Apply BALANCED standards: catch blunders, ignore minor
imperfections. When a flaw is small and the image still looks believable,
PASS it. BUT be strict and careful about hands, palms, arms and legs.

STEP 1 — DESCRIBE THE LIMBS (you must fill these in honestly and in detail;
this description is mandatory and comes BEFORE any judgement):
- limb_description.hands: describe every hand you can see. For EACH hand
  state: which arm/wrist it is on, how many palms are on that wrist (a
  normal wrist has exactly ONE palm), whether you see any extra palm,
  doubled palm, or extra hand-shaped mass attached to it.
- limb_description.arms: describe the arms — how many, where each connects
  to the body, and whether any arm is attached at an impossible place or
  bends impossibly.
- limb_description.legs: describe the legs — how many, and whether any leg
  is doubled, fused, or attached wrongly.

STEP 2 — JUDGE. Using your STEP 1 descriptions, answer each question. If
your description above mentions an extra/doubled palm or hand-mass, then
has_malformed_hand MUST be true.

ANATOMY:
1. person_count: how many distinct real people are visible? (must be 1)
2. has_extra_limbs: more than 2 arms, OR more than 2 legs, OR more than 2
   hands clearly visible? Ignore a limb partially hidden or cropped.
   (expected: false)
3. has_malformed_hand: based on STEP 1 — does any ONE hand have a doubled
   palm, a second palm on the same wrist, an extra hand-shaped mass, or a
   hand attached at an impossible place? (expected: false)
4. has_malformed_arm_or_leg: based on STEP 1 — is any arm or leg doubled,
   fused, bent in a physically impossible way, or attached at an impossible
   place on the body? (expected: false)
5. has_blatant_finger_blunder: does any ONE hand show 7 OR MORE clearly
   visible fingers, OR fingers fused into a single mass? If fingers are
   simply hidden behind the product, that is NORMAL — answer false. Do NOT
   flag low finger counts or slight curl. (expected: false)
6. face_grossly_distorted: is the face severely distorted, melted, doubled,
   or missing? Ignore minor asymmetry or soft focus. (expected: false)

PLACEMENT (logical, not perfect):
7. placement_illogical: is the way the person holds or is positioned with
   the product clearly ILLOGICAL — product floats unsupported, the grip is
   one no real hand could make, or the body pose is physically absurd? A
   slightly awkward but physically possible pose is FINE. (expected: false)

PRODUCT:
8. product_visible: is the Alluvi product box visible? (expected: true)
9. multiple_distinct_products: are there 2 OR MORE separate copies of the
   product box? (a mirror reflection does NOT count) (expected: false)
10. product_shape_broken: is the box warped, melted, or bent into a
    non-rectangular impossible shape? Minor perspective/angle is FINE.
    (expected: false)
11. brand_name_legible: the box should show "ALLUVI". Is "ALLUVI" at least
    90% legible — a viewer clearly reads it as ALLUVI (not "ARLUVI",
    "ALUUVI", or unreadable)? This is the STRICT check. (expected: true)
12. product_name_resemblance: the box should show "TIRZEPATIDE". Does the
    rendered text resemble "TIRZEPATIDE" by at least ~70% — recognisable
    even if a letter or two is off? (expected: true)
13. box_theme_ok: is the box a rectangular box with the correct colour
    theme (white base, blue wave graphic, green seal) at roughly 80%+?
    Exact content not required. (expected: true)

Then provide:
- specific_issues: short list of ONLY the real defects found. Each item a
  SHORT imperative phrase (e.g. "doubled palm on the left hand", "extra
  hand-mass on the right wrist", "brand name ALLUVI garbled"). Empty list
  if the image is acceptable.
- overall_recommendation: "use", "regenerate", or "discard"
- confidence: 0.0 to 1.0

Respond with EXACTLY this JSON, no extra keys:
{
  "limb_description": {
    "hands": "<plain-words description of every hand>",
    "arms": "<plain-words description of the arms>",
    "legs": "<plain-words description of the legs>"
  },
  "person_count": <int>,
  "has_extra_limbs": <bool>,
  "has_malformed_hand": <bool>,
  "has_malformed_arm_or_leg": <bool>,
  "has_blatant_finger_blunder": <bool>,
  "face_grossly_distorted": <bool>,
  "placement_illogical": <bool>,
  "product_visible": <bool>,
  "multiple_distinct_products": <bool>,
  "product_shape_broken": <bool>,
  "brand_name_legible": <bool>,
  "product_name_resemblance": <bool>,
  "box_theme_ok": <bool>,
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
    Run BALANCED QC validation with describe-then-judge limb checking.

    Returns a dict with `passed`, `score`, `checks`, `issues`,
    `recommendation`, `confidence`, `model`, `error`, `raw_vlm_response`.
    `issues` is a list of SHORT imperative defect phrases — the retry loop
    appends them to the Stage 2 prompt as a clean "AVOID:" line.
    """
    if not image_path.exists():
        return _error_result(
            f"image file not found: {image_path}", scenario_id=scenario_id)

    print(f"[qc_validator] {scenario_id}: running QC via {QC_MODEL}...")

    try:
        image_data, media_type = _load_image_base64(image_path)
    except Exception as e:
        return _error_result(
            f"failed to load image: {type(e).__name__}: {e}",
            scenario_id=scenario_id)

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
            scenario_id=scenario_id)

    try:
        raw_text = response.content[0].text
        result = validate_json_output(
            raw_text, required_keys=["overall_recommendation"])
    except JSONSanityError as e:
        return _error_result(
            f"QC response JSON parse failed: {e}", scenario_id=scenario_id)

    decision = _score_qc_result(result)

    status = "PASS" if decision["passed"] else "FAIL"
    print(
        f"[qc_validator] {scenario_id}: {status} "
        f"(score={decision['score']:.2f}, issues={len(decision['issues'])}, "
        f"rec={decision['recommendation']})"
    )
    for issue in decision["issues"]:
        print(f"  - {issue}")

    return decision


# ─── Internal helpers ─────────────────────────────────────────────────────

def _load_image_base64(image_path: Path) -> tuple[str, str]:
    """Load image, return (base64_string, media_type)."""
    suffix = image_path.suffix.lower()
    media_type = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
    }.get(suffix, "image/jpeg")
    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, media_type


def _score_qc_result(result: dict) -> dict[str, Any]:
    """
    BALANCED scoring. Any hard-fail check = regenerate. Each fail also adds a
    SHORT imperative correction phrase to `issues` for the retry "AVOID:"
    line — phrased as a plain instruction, with no mention of attempts or
    that a previous image had the defect.
    """
    defects: list[str] = []

    # ── anatomy ──────────────────────────────────────────────────────
    person_count = result.get("person_count")
    if person_count is not None and person_count != 1:
        defects.append(f"show exactly one person (not {person_count})")

    if result.get("has_extra_limbs"):
        defects.append("show exactly two arms, two hands and two legs — no extra limbs")

    if result.get("has_malformed_hand"):
        defects.append("each wrist has exactly one clean natural hand with one palm — no doubled or extra palms")

    if result.get("has_malformed_arm_or_leg"):
        defects.append("keep every arm and leg natural, separate, and attached at the correct place")

    if result.get("has_blatant_finger_blunder"):
        defects.append("render normal natural hands with a normal number of fingers")

    if result.get("face_grossly_distorted"):
        defects.append("keep the face clean, natural and undistorted")

    # ── placement ────────────────────────────────────────────────────
    if result.get("placement_illogical"):
        defects.append("make the product hold and the body pose natural and physically plausible")

    # ── product ──────────────────────────────────────────────────────
    if not result.get("product_visible", True):
        defects.append("the Alluvi product box must be clearly visible in the scene")

    if result.get("multiple_distinct_products"):
        defects.append("show exactly ONE Alluvi product box")

    if result.get("product_shape_broken"):
        defects.append("keep the product a clean undistorted rectangular box")

    if not result.get("brand_name_legible", True):
        defects.append('the box must clearly read "ALLUVI"')

    if not result.get("product_name_resemblance", True):
        defects.append('the box text must clearly resemble "TIRZEPATIDE"')

    if not result.get("box_theme_ok", True):
        defects.append("the box is white with a blue wave graphic and a green seal")

    # ── VLM's own free-text issues (merged, deduped) ──────────────────
    extra_issues = result.get("specific_issues") or []
    if not isinstance(extra_issues, list):
        extra_issues = [str(extra_issues)]

    passed = len(defects) == 0
    score = max(0.0, 1.0 - (len(defects) * 0.15))

    seen: set = set()
    all_issues: list[str] = []
    for src in (defects, extra_issues):
        for item in src:
            s = str(item).strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                all_issues.append(s)

    rec = result.get("overall_recommendation", "")
    if rec not in ("use", "regenerate", "discard"):
        rec = "use" if passed else "regenerate"

    return {
        "passed": passed,
        "score": round(score, 3),
        "checks": {
            "person_count": result.get("person_count"),
            "has_extra_limbs": result.get("has_extra_limbs"),
            "has_malformed_hand": result.get("has_malformed_hand"),
            "has_malformed_arm_or_leg": result.get("has_malformed_arm_or_leg"),
            "has_blatant_finger_blunder": result.get("has_blatant_finger_blunder"),
            "face_grossly_distorted": result.get("face_grossly_distorted"),
            "placement_illogical": result.get("placement_illogical"),
            "product_visible": result.get("product_visible"),
            "multiple_distinct_products": result.get("multiple_distinct_products"),
            "product_shape_broken": result.get("product_shape_broken"),
            "brand_name_legible": result.get("brand_name_legible"),
            "product_name_resemblance": result.get("product_name_resemblance"),
            "box_theme_ok": result.get("box_theme_ok"),
        },
        "limb_description": result.get("limb_description"),
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
    """Build a result when the QC call itself fails — treat as pass (do not
    punish the image for a QC-infrastructure outage)."""
    print(f"[qc_validator] {scenario_id}: QC ERROR — {error_message}")
    return {
        "passed": True,
        "score": 0.5,
        "checks": {},
        "limb_description": None,
        "issues": [f"QC validation failed (treating as pass): {error_message}"],
        "recommendation": "use",
        "confidence": 0.0,
        "error": error_message,
        "model": QC_MODEL,
        "raw_vlm_response": None,
    }