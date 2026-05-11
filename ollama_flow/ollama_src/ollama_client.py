"""
ollama_flow/ollama_src/ollama_client.py — thin client for local Ollama.

Calls Ollama's POST /api/generate endpoint. Used by the Ollama-mode prompt
builders as a drop-in replacement for the Anthropic client.

Cost: $0 per call (runs locally on your machine).
Latency: typically 5-30s per call on a recent GPU/Apple Silicon, varies
by model size and prompt length.

Quality NOTE: a 7B model like qwen2.5:7b is significantly weaker than
Claude Opus 4.7 at:
  - producing strictly valid JSON
  - following multi-page system prompts with many rules
  - staying within word budgets
  - copying long descriptors verbatim

JSON validity is handled by the shared `src.json_utils.validate_json_output`
function — same code that the Claude builders use. It defensively handles
markdown fences, leading/trailing prose, and trailing commas. It cannot
rescue a fundamentally bad response — only graceful failure with a clear
error message.

When the Step 2 master prompt's clauses are dropped or the Step 1 word
budget is blown, the outputs to fal will still go through — they'll just
produce lower-quality images than the Claude builders. The pipeline runs
end-to-end either way; that's the point of iteration mode.
"""

import os
import httpx
from pathlib import Path
from typing import Any
from dotenv import load_dotenv


# Load .env from the ollama_flow folder (sibling to this ollama_src/)
_OLLAMA_FLOW_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_OLLAMA_FLOW_ROOT / ".env")


DEFAULT_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
DEFAULT_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns an error."""


def generate(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    host: str | None = None,
    timeout: float | None = None,
    options: dict | None = None,
) -> str:
    """
    Call Ollama's /api/generate endpoint. Returns the raw response text.

    Args:
        prompt: the user message
        system: the system prompt (master prompt text)
        model: Ollama model id (default: qwen2.5:7b)
        host: Ollama host URL (default: http://localhost:11434)
        timeout: request timeout in seconds (default: 180)
        options: Ollama generation options dict
                 (temperature, top_p, num_predict, etc.)

    Returns:
        Raw response text from the model.
    """
    host = host or DEFAULT_HOST
    model = model or DEFAULT_MODEL
    timeout = timeout or DEFAULT_TIMEOUT

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system
    if options:
        payload["options"] = options

    url = host.rstrip("/") + "/api/generate"

    try:
        response = httpx.post(url, json=payload, timeout=timeout)
    except httpx.ConnectError as e:
        raise OllamaError(
            f"Cannot connect to Ollama at {host}. "
            f"Is Ollama running? Try `ollama serve` in a separate terminal. "
            f"Underlying error: {e}"
        ) from e
    except httpx.ReadTimeout as e:
        raise OllamaError(
            f"Ollama at {host} did not respond within {timeout}s. "
            f"The model may be loading for the first time, or the prompt "
            f"may be too long. Try increasing OLLAMA_TIMEOUT_SECONDS in .env. "
            f"Underlying error: {e}"
        ) from e

    if response.status_code != 200:
        # Ollama returns 404 if the model isn't pulled, 500 on internal errors
        body = response.text[:500]
        if response.status_code == 404:
            raise OllamaError(
                f"Ollama returned 404. The model {model!r} is probably not "
                f"pulled. Run `ollama pull {model}` and retry. "
                f"Response body: {body}"
            )
        raise OllamaError(
            f"Ollama returned HTTP {response.status_code}. Body: {body}"
        )

    data = response.json()
    return data.get("response", "")


def generate_json(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    host: str | None = None,
    timeout: float | None = None,
    options: dict | None = None,
    required_keys: list[str] | None = None,
) -> dict:
    """
    Call generate() and parse the response as JSON, with defensive extraction
    and optional required-key validation.

    The shared `src.json_utils.validate_json_output` does the heavy lifting:
      1. Strip markdown fences (```json ... ```)
      2. Find the outermost balanced { ... } object
      3. Strip trailing commas before } or ]
      4. Validate required_keys are present and non-empty
      5. Raise JSONSanityError with a snippet of the raw text if anything fails

    Args:
        prompt, system, model, host, timeout, options:
            Forwarded to generate().
        required_keys:
            Optional list of top-level keys that must be present in the
            parsed JSON and non-empty. Step 1 builder passes
            ["step_1_image_prompt"]; Step 2 builder passes ["step_2_image_prompt"].

    Raises:
        OllamaError: if the Ollama server call itself fails
        JSONSanityError: if the response cannot be parsed as valid JSON
                         or required keys are missing/empty
    """
    # Import here (not at module top) so test/dev environments without
    # the parent repo on sys.path can still import this module's other
    # functions for unit testing in isolation.
    from src.json_utils import validate_json_output

    raw = generate(
        prompt,
        system=system,
        model=model,
        host=host,
        timeout=timeout,
        options=options,
    )
    return validate_json_output(raw, required_keys=required_keys)