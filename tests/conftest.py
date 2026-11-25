"""
Shared test fixtures.

Tests never touch the network or real API keys: the search API, website
fetching and the LLM are replaced with fakes, so results are deterministic.
"""

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app import main
from app.config import get_settings
from app.models import SearchResult
from app.services import profile_extractor, query_service, search_service
from app.services.llm_service import get_llm_service

# Fake Google results, chosen by which site filter the query contains.
FAKE_RESULTS = {
    "site:instagram.com": [
        SearchResult("Jane Doe (@janedoe) • Instagram photos and videos", "https://www.instagram.com/janedoe/",
                     "DC football fan. Washington DC | jane@doe.io", ""),
        SearchResult("John Smith (@jsmith) • Instagram photos and videos", "https://www.instagram.com/jsmith/",
                     "Travel photos and coffee", ""),  # irrelevant: no football evidence
    ],
    "site:x.com": [
        SearchResult("Jane Doe (@janedoe) / X", "https://x.com/janedoe",
                     "Football is life. Commanders season ticket holder", ""),
    ],
    "site:linkedin.com": [
        SearchResult("Mike Ross - Football Coach - DC United | LinkedIn", "https://www.linkedin.com/in/mike-ross",
                     "Washington, District of Columbia. Youth football coach.", ""),
    ],
    "site:facebook.com": [],
    "site:threads": [],
    "site:youtube.com": [
        SearchResult("Top 10 football channels", "https://www.youtube.com/watch?v=abc", "list", ""),
    ],
    "-site:linkedin.com": [  # Google websites
        SearchResult("Top 10 Football Bars in Washington DC", "https://blog.example.org/top-10", "list", ""),
        SearchResult("Capital Kickers | Washington DC Football Club", "https://capitalkickers.org/", "Football club", ""),
    ],
    "whatsapp": [
        SearchResult("Capital Kickers | Football Club", "https://capitalkickers.org/join",
                     "Join DC's friendliest football club. WhatsApp: +1 202 456 1111", ""),
    ],
    "email": [],
}

FAKE_PAGE = """<html><head><meta property="og:site_name" content="Capital Kickers"></head>
<body><a href="mailto:hello@capitalkickers.org">mail</a>
<a href="https://www.instagram.com/capitalkickers/">Instagram</a></body></html>"""


def fake_results_for(query: str) -> list[SearchResult]:
    # More specific markers first ("-site:linkedin.com" contains "site:linkedin.com").
    for marker in ("-site:linkedin.com", "whatsapp", "email", "site:instagram.com", "site:x.com",
                   "site:linkedin.com", "site:facebook.com", "site:threads", "site:youtube.com"):
        if marker in query:
            return [SearchResult(r.title, r.link, r.snippet, query) for r in FAKE_RESULTS[marker]]
    return []


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "serper_api_key", SecretStr("test-key"))
    monkeypatch.setattr(settings, "youtube_api_key", None)
    monkeypatch.setattr(settings, "anthropic_api_key", None)
    monkeypatch.setattr(get_llm_service(), "_api_key", None)  # LLM off unless a test enables a fake
    monkeypatch.setattr(main.rate_limiter, "max_requests", 1000)
    query_service._cache.clear()
    search_service._cache.clear()

    async def fake_google_search(query, country_code=None):
        return fake_results_for(query)

    async def fake_fetch(url):
        return (url, FAKE_PAGE) if "capitalkickers.org" in url else None

    monkeypatch.setattr(search_service, "google_search", fake_google_search)
    monkeypatch.setattr(profile_extractor, "fetch_public_html", fake_fetch)
    yield


@pytest.fixture
def client():
    with TestClient(main.app) as test_client:
        yield test_client
