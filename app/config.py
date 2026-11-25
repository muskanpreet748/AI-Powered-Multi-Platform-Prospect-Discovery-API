"""
Application settings.

All configuration comes from environment variables (or a local `.env` file).
API keys are stored as `SecretStr`, so they are hidden ("**********") if a
settings object is ever printed or logged by accident.
"""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM - "anthropic" (recommended for production) or "gemini" (free tier)
    llm_provider: Literal["anthropic", "gemini"] = "anthropic"
    llm_model: str | None = None  # empty = the provider's default model
    anthropic_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    llm_timeout_seconds: float = 30.0

    # Discovery sources
    serper_api_key: SecretStr | None = None
    youtube_api_key: SecretStr | None = None

    # Discovery limits (keep API usage and cost predictable)
    max_results_per_query: int = 10
    max_queries_per_platform: int = 2
    max_websites_to_enrich: int = 6
    max_prospects: int = 50
    max_llm_relevance_candidates: int = 40

    # Timeouts, caching and rate limits
    platform_timeout_seconds: float = 25.0
    http_timeout_seconds: float = 10.0
    cache_ttl_seconds: int = 3600
    search_concurrency: int = 4
    rate_limit_per_minute: int = 10

    log_level: str = "INFO"

    @property
    def llm_api_key(self) -> SecretStr | None:
        return self.anthropic_api_key if self.llm_provider == "anthropic" else self.gemini_api_key

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key and self.llm_api_key.get_secret_value())

    @property
    def search_enabled(self) -> bool:
        return bool(self.serper_api_key and self.serper_api_key.get_secret_value())

    @property
    def youtube_api_enabled(self) -> bool:
        return bool(self.youtube_api_key and self.youtube_api_key.get_secret_value())


@lru_cache
def get_settings() -> Settings:
    """Return one shared Settings object (created on first use)."""
    return Settings()
