"""
src/json_utils.py — shared JSON sanity-check + extraction utilities.

Used by both the Claude flow (src/step_1_prompt_builder.py, step_2_*.py)
and the Ollama flow (ollama_flow/ollama_src/ollama_client.py).

The single public function `validate_json_output(text, required_keys)`:
  1. DEFENSIVELY EXTRACTS JSON from raw LLM output, handling the most
     common failure modes:
       - Markdown code fences (```json ... ``` or ``` ... ```)
       - Leading explanation prose ("Here is the JSON: { ... }")
       - Trailing prose after the JSON closes
       - Trailing commas before } or ] (Python-valid, JSON-invalid)
  2. VALIDATES required top-level keys are present and non-empty.
  3. RAISES `ValueError` with a clear, debuggable message if anything
     fails — including a snippet of the raw response for log inspection.

Both Opus and qwen2.5:7b occasionally produce malformed JSON; this guard
catches it cleanly at the parse boundary so failures show up as
'step_1_prompt' or 'step_2_prompt' DB rows with a useful error message,
rather than crashing the run.
"""

import json
import re


class JSONSanityError(ValueError):
    """Raised when LLM output cannot be parsed as JSON or fails schema check.

    Subclass of ValueError so existing `except ValueError` blocks still catch
    it, while letting more specific handlers distinguish it from other errors.
    """


def validate_json_output(
    text: str,
    *,
    required_keys: list[str] | None = None,
) -> dict:
    """
    Extract and validate JSON from raw LLM output.

    Args:
        text: raw response text from the LLM
        required_keys: list of top-level keys that must be present in the
                       parsed JSON AND non-empty (None, "", [], {} all fail).
                       Pass None or [] to skip key validation.

    Returns:
        The parsed JSON object as a dict.

    Raises:
        JSONSanityError: if extraction fails or required keys are missing.
    """
    if not isinstance(text, str):
        raise JSONSanityError(
            f"LLM output is not a string (got {type(text).__name__})"
        )

    # Step 1: defensive extraction
    try:
        data = _extract_json(text)
    except JSONSanityError:
        raise
    except Exception as e:
        # Catch any unexpected internal error from the extractor
        raise JSONSanityError(
            f"JSON extraction crashed: {type(e).__name__}: {e}"
        ) from e

    if not isinstance(data, dict):
        raise JSONSanityError(
            f"LLM output parsed but is not a JSON object "
            f"(got {type(data).__name__}: {data!r:.200s}...)"
        )

    # Step 2: validate required keys
    if required_keys:
        missing = []
        empty = []
        for key in required_keys:
            if key not in data:
                missing.append(key)
            elif not data[key]:
                # Treat None, "", [], {} as empty
                empty.append(key)

        if missing or empty:
            errs = []
            if missing:
                errs.append(f"missing keys: {missing}")
            if empty:
                errs.append(f"empty keys: {empty}")
            present = sorted(data.keys())
            raise JSONSanityError(
                "LLM output parsed but failed schema check — "
                + "; ".join(errs)
                + f"; keys present: {present}"
            )

    return data


# ──────────────────────────────────────────────────────────────────────────
# Internal helpers — defensive JSON extraction
# ──────────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """
    Robustly extract a JSON object from raw model output.

    Strategy (try each in order, return on first success):
      1. Strip common markdown fences and parse directly
      2. Find the substring from the first '{' to the last matching '}'
      3. Strip trailing commas before } or ]
      4. Give up and raise JSONSanityError with the raw text snippet
    """
    cleaned = text.strip()
    if not cleaned:
        raise JSONSanityError("LLM output is empty after stripping whitespace")

    # Strategy 1: strip code fences
    fence_stripped = _strip_code_fences(cleaned)
    try:
        return json.loads(fence_stripped)
    except json.JSONDecodeError:
        pass

    # Strategy 2: find the outermost balanced { ... } object
    balanced = _find_outermost_object(cleaned)
    if balanced is not None:
        try:
            return json.loads(balanced)
        except json.JSONDecodeError:
            # Strategy 3: strip trailing commas
            try:
                return json.loads(_strip_trailing_commas(balanced))
            except json.JSONDecodeError:
                pass

    # Strategy 4: give up with a useful error
    snippet = cleaned[:500] + ("..." if len(cleaned) > 500 else "")
    raise JSONSanityError(
        f"Could not extract valid JSON from LLM output. "
        f"Raw output (first 500 chars):\n{snippet}"
    )


def _strip_code_fences(text: str) -> str:
    """Strip ```json or ``` fences from the start/end."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _find_outermost_object(text: str) -> str | None:
    """
    Walk the string and find the first '{' through its matching '}'.
    Respects nested objects and string literals (won't be confused by
    braces inside strings).
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def _strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before } or ]. Naive but works for most cases."""
    return re.sub(r",(\s*[}\]])", r"\1", text)