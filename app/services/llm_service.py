"""
LLM service - the only place that talks to the Anthropic (Claude) API.

We use "structured outputs": we hand the SDK a Pydantic model, the API is
constrained to return JSON matching that schema, and the SDK validates it and
gives us back a Python object (`response.parsed_output`). So the rest of the
code never has to parse free-form LLM text.

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

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """The LLM could not produce a usable answer (callers fall back to rules)."""


class LLMService:
    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.llm_model
        self.client: anthropic.AsyncAnthropic | None = None
        if settings.llm_enabled:
            self.client = anthropic.AsyncAnthropic(
                api_key=settings.anthropic_api_key.get_secret_value(),
                timeout=settings.llm_timeout_seconds,
                max_retries=2,  # SDK retries 429/5xx/connection errors with backoff
            )

    @property
    def enabled(self) -> bool:
        return self.client is not None

    async def structured(self, *, system: str, user: str, output_model: type[T], max_tokens: int = 2000) -> T:
        """Send one prompt and return the answer validated against `output_model`."""
        if self.client is None:
            raise LLMError("LLM is not configured (ANTHROPIC_API_KEY missing)")

        started = time.perf_counter()
        try:
            response = await self.client.messages.parse(
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

        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "LLM %s | %s | %.0f ms | input=%d output=%d tokens",
            output_model.__name__, self.model, elapsed_ms,
            response.usage.input_tokens, response.usage.output_tokens,
        )

        if response.stop_reason == "refusal":
            raise LLMError("LLM declined the request")
        if response.stop_reason == "max_tokens":
            raise LLMError("LLM output was cut off (max_tokens)")
        if response.parsed_output is None:
            raise LLMError("LLM returned no structured output")
        return response.parsed_output


@lru_cache
def get_llm_service() -> LLMService:
    return LLMService()
