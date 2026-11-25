"""
Search service - Google search results through the Serper.dev API.

Why Serper? Google's own Custom Search JSON API is closed to new customers
(and shuts down on 1 Jan 2027). Serper is an authorized, paid SERP API that
returns Google's organic results as JSON, so we never scrape google.com.

Features: response caching, a concurrency limit (so we never flood the API),
and automatic retries on 429/5xx (see `request_with_retries`).
"""

import asyncio
import logging

from app.config import get_settings
from app.models import SearchResult
from app.utils.cache import TTLCache
from app.utils.http_client import ExternalAPIError, request_with_retries

logger = logging.getLogger(__name__)

SERPER_URL = "https://google.serper.dev/search"

_cache = TTLCache(ttl_seconds=get_settings().cache_ttl_seconds)
_semaphore = asyncio.Semaphore(get_settings().search_concurrency)


class SearchError(Exception):
    pass


async def google_search(query: str, country_code: str | None = None) -> list[SearchResult]:
    settings = get_settings()
    if not settings.search_enabled:
        raise SearchError("SERPER_API_KEY is not configured")

    payload: dict = {"q": query, "num": settings.max_results_per_query}
    if country_code:
        payload["gl"] = country_code.lower()  # localise results to the target country

    cache_key = f"{query}|{payload.get('gl')}|{payload['num']}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    async with _semaphore:
        try:
            response = await request_with_retries(
                "POST", SERPER_URL, json=payload,
                headers={"X-API-KEY": settings.serper_api_key.get_secret_value()},
            )
        except ExternalAPIError as exc:
            raise SearchError(f"search API unreachable ({exc})") from exc

    if response.status_code in (401, 403):
        raise SearchError("search API key was rejected")
    if response.status_code == 429:
        raise SearchError("search API rate limit / credits exhausted")
    if response.status_code != 200:
        raise SearchError(f"search API returned HTTP {response.status_code}")

    try:
        organic = response.json().get("organic", [])
    except ValueError as exc:
        raise SearchError("search API returned invalid JSON") from exc

    results = [
        SearchResult(
            title=str(item.get("title", "")),
            link=str(item.get("link", "")),
            snippet=str(item.get("snippet", "")),
            query=query,
        )
        for item in organic
        if isinstance(item, dict) and item.get("link")
    ]
    _cache.set(cache_key, results)
    return results


async def search_many(queries: list[str], country_code: str | None) -> list[SearchResult]:
    """
    Run several queries concurrently. If some fail we keep the others; only if
    *all* fail do we raise, so the platform is reported as failed.
    """
    outcomes = await asyncio.gather(
        *(google_search(q, country_code) for q in queries), return_exceptions=True
    )
    results: list[SearchResult] = []
    errors: list[BaseException] = []
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            errors.append(outcome)
        else:
            results.extend(outcome)
    if errors and not results:
        raise errors[0]
    for err in errors:
        logger.warning("One search query failed: %s", err)
    return results
