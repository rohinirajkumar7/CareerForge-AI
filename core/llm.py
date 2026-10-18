"""One place that talks to the language model.

Every agent goes through `generate_json`, so retries, JSON parsing, schema
validation, the system instruction and error messages are handled once.
The transport is swappable, which keeps tests and offline evaluation free of
network calls and API keys.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from functools import lru_cache
from typing import TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from security.injection_guard import SYSTEM_RULES

load_dotenv()

logger = logging.getLogger("careerforge.llm")

DEFAULT_MODEL = "gemini-3.5-flash-lite"
TRANSIENT_CODES = {429, 500, 502, 503, 504}
BACKOFF_SECONDS = (1.0, 2.0, 4.0)

T = TypeVar("T", bound=BaseModel)

# transport(system_instruction, prompt, temperature) -> raw model text
Transport = Callable[[str, str, float], str]


class LLMError(RuntimeError):
    """Raised with a message that is safe to show to end users."""


class TransientLLMError(RuntimeError):
    """A temporary failure (rate limit, overload) that is worth retrying."""


def model_name() -> str:
    return os.getenv("GEMINI_MODEL", DEFAULT_MODEL)


@lru_cache(maxsize=1)
def _client():
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY is not set. Add it to your .env file.")
    return genai.Client(api_key=api_key)


def gemini_transport(system: str, prompt: str, temperature: float) -> str:
    """Default transport: Gemini in JSON mode."""
    from google.genai import errors, types

    client = _client()
    try:
        response = client.models.generate_content(
            model=model_name(),
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                temperature=temperature,
            ),
        )
    except errors.APIError as error:
        if getattr(error, "code", None) in TRANSIENT_CODES:
            raise TransientLLMError(str(error)) from error
        raise LLMError("The AI service rejected the request. Check your API key and model name.") from error
    return response.text or ""


_transport: Transport = gemini_transport


def set_transport(transport: Transport | None) -> None:
    """Replace the transport (tests, offline evaluation). None restores Gemini."""
    global _transport
    _transport = transport or gemini_transport


def _strip_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def generate_json(
    prompt: str,
    schema: type[T],
    *,
    system: str = SYSTEM_RULES,
    temperature: float = 0.2,
    retries: int = 2,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call the model and return a validated `schema` instance.

    Retries on transient service errors and on malformed output.
    Raises LLMError with a user-safe message when all attempts fail.
    """
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            raw = _transport(system, prompt, temperature)
            return schema.model_validate(json.loads(_strip_fences(raw)))
        except LLMError:
            raise
        except TransientLLMError as error:
            last_error = error
            logger.warning("transient LLM error (attempt %d): %s", attempt + 1, error)
            if attempt < retries:
                sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
        except (json.JSONDecodeError, ValidationError) as error:
            last_error = error
            logger.warning("invalid model output (attempt %d): %s", attempt + 1, type(error).__name__)
    raise LLMError("The AI service did not return a usable answer. Please try again in a moment.") from last_error
