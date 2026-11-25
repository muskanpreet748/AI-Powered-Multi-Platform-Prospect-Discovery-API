"""
LLM service - the only place that talks to an LLM API.

Two interchangeable providers (chosen with LLM_PROVIDER):
  * "anthropic" - Claude via the official Anthropic SDK (recommended for production)
  * "gemini"    - Google Gemini via its REST API (has a free tier)

Both use "structured outputs": we send a JSON schema generated from a Pydantic
model, the API is constrained to return JSON matching it, and we validate the
answer back into that Pydantic model. The rest of the code just calls
`structured(...)` and never needs to know which provider is used.

The LLM is used only for *understanding and classifying text*. It never
produces URLs, names, emails or phone numbers that end up in the response.
"""

import logging
import time
from functools import lru_cache
from typing import TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.utils.http_client import ExternalAPIError, request_with_retries

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODELS = {"anthropic": "claude-haiku-5-5", "gemini": "gemini-3.5-flash-lite"}
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class LLMError(Exception):
    """The LLM could not produce a usable answer (callers fall back to rules)."""


class LLMService:
    def __init__(self) -> None:
        settings = get_settings()
        self.provider = settings.llm_provider
        self.model = settings.llm_model or DEFAULT_MODELS[self.provider]
        self.timeout = settings.llm_timeout_seconds
        self._api_key = settings.llm_api_key.get_secret_value() if settings.llm_enabled else None
        self._anthropic: anthropic.AsyncAnthropic | None = None
        if self._api_key and self.provider == "anthropic":
            self._anthropic = anthropic.AsyncAnthropic(
                api_key=self._api_key,
                timeout=self.timeout,
                max_retries=2,  # SDK retries 429/5xx/connection errors with backoff
            )

    @property
    def enabled(self) -> bool:
        return self._api_key is not None

    async def structured(self, *, system: str, user: str, output_model: type[T], max_tokens: int = 2000) -> T:
        """Send one prompt and return the answer validated against `output_model`."""
        if not self.enabled:
            raise LLMError("LLM is not configured (API key missing)")

        started = time.perf_counter()
        if self.provider == "anthropic":
            result, input_tokens, output_tokens = await self._call_anthropic(system, user, output_model, max_tokens)
        else:
            result, input_tokens, output_tokens = await self._call_gemini(system, user, output_model, max_tokens)

        logger.info(
            "LLM %s | %s | %.0f ms | input=%d output=%d tokens",
            output_model.__name__, self.model, (time.perf_counter() - started) * 1000,
            input_tokens, output_tokens,
        )
        return result

    # ------------------------------------------------------------------ #
    # Anthropic (Claude) - official SDK
    # ------------------------------------------------------------------ #
    async def _call_anthropic(self, system: str, user: str, output_model: type[T], max_tokens: int
                              ) -> tuple[T, int, int]:
        try:
            response = await self._anthropic.messages.parse(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=output_model,
                # Simple extraction/classification: low effort = faster and cheaper.
                output_config={"effort": "low"},
            )
        except anthropic.AuthenticationError as exc:
            raise LLMError("invalid Anthropic API key") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("LLM rate limit reached") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"LLM API error (status {exc.status_code})") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("could not reach the LLM API") from exc
        except (ValidationError, ValueError) as exc:  # malformed JSON or schema mismatch
            raise LLMError("LLM returned data that did not match the schema") from exc

        if response.stop_reason == "refusal":
            raise LLMError("LLM declined the request")
        if response.stop_reason == "max_tokens":
            raise LLMError("LLM output was cut off (max_tokens)")
        if response.parsed_output is None:
            raise LLMError("LLM returned no structured output")
        return response.parsed_output, response.usage.input_tokens, response.usage.output_tokens

    # ------------------------------------------------------------------ #
    # Google Gemini - REST API (generateContent with a JSON schema)
    # ------------------------------------------------------------------ #
    async def _call_gemini(self, system: str, user: str, output_model: type[T], max_tokens: int
                           ) -> tuple[T, int, int]:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": output_model.model_json_schema(),
                "maxOutputTokens": max_tokens,
            },
        }
        try:
            response = await request_with_retries(
                "POST", f"{GEMINI_URL}/{self.model}:generateContent",
                json=body,
                headers={"x-goog-api-key": self._api_key},  # header, not URL -> never logged
                timeout=self.timeout,
                attempts=4,  # free-tier models are often briefly overloaded (HTTP 503)
            )
        except ExternalAPIError as exc:
            raise LLMError("could not reach the Gemini API") from exc

        if response.status_code in (401, 403):
            raise LLMError("invalid Gemini API key")
        if response.status_code == 429:
            raise LLMError("LLM rate limit reached")
        if response.status_code != 200:
            raise LLMError(f"LLM API error (status {response.status_code})")

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError("Gemini returned invalid JSON") from exc

        candidates = data.get("candidates") or []
        if not candidates:
            raise LLMError("LLM returned no output (request may have been blocked)")
        finish_reason = candidates[0].get("finishReason")
        if finish_reason == "MAX_TOKENS":
            raise LLMError("LLM output was cut off (max_tokens)")
        if finish_reason not in (None, "STOP"):
            raise LLMError(f"LLM declined the request ({finish_reason})")

        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        try:
            result = output_model.model_validate_json(text)
        except ValidationError as exc:
            raise LLMError("LLM returned data that did not match the schema") from exc

        usage = data.get("usageMetadata") or {}
        output_tokens = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
        return result, usage.get("promptTokenCount", 0), output_tokens


@lru_cache
def get_llm_service() -> LLMService:
    return LLMService()
