"""Tests for the prospect discovery API and its building blocks."""

import asyncio

import pytest

from app.models import ProspectGroup, RawProfile
from app.services import search_service
from app.services.deduplication_service import consolidate
from app.services.llm_service import LLMError, get_llm_service
from app.services.query_service import rule_based_understanding, understand_query
from app.services.relevance_service import CandidateVerdict, combine, rule_signals
from app.utils.http_client import UnsafeURLError, assert_public_url
from app.utils.normalizers import clean_profile_name, detect_profile
from app.utils.validators import extract_whatsapp_numbers, validate_email, validate_phone

VALID_PLATFORMS = {"Google", "LinkedIn", "X", "YouTube", "Instagram", "WhatsApp", "Facebook", "Email", "Threads", "Phone"}


# --------------------------------------------------------------------------- #
# Validators & normalizers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw, expected", [
    ("Jane@Acme.co", "jane@acme.co"),
    ("mailto:info@shop.ae?subject=hi", "info@shop.ae"),
    ("logo@2x.png", None),          # image file name, not an email
    ("name@example.com", None),     # placeholder
    ("not-an-email", None),
])
def test_validate_email(raw, expected):
    assert validate_email(raw) == expected


def test_validate_phone_and_whatsapp():
    assert validate_phone("(202) 456-1111", "US") == "+12024561111"
    assert validate_phone("12345", "US") is None
    assert extract_whatsapp_numbers("WhatsApp: +971 50 123 4567", "AE") == {"+971501234567"}
    assert extract_whatsapp_numbers("Call +971 50 123 4567", "AE") == set()  # not labelled WhatsApp


@pytest.mark.parametrize("url, platform, canonical", [
    ("https://uk.linkedin.com/in/Jane-Doe?trk=1", "LinkedIn", "https://www.linkedin.com/in/jane-doe"),
    ("https://www.instagram.com/jane.doe/", "Instagram", "https://www.instagram.com/jane.doe/"),
    ("https://twitter.com/JaneDoe/status/123", "X", "https://x.com/janedoe"),
    ("https://www.threads.net/@jane", "Threads", "https://www.threads.com/@jane"),
    ("https://www.youtube.com/@JaneTV", "YouTube", "https://www.youtube.com/@janetv"),
    ("https://m.facebook.com/acmegrill/", "Facebook", "https://www.facebook.com/acmegrill"),
    ("https://wa.me/971501234567", "WhatsApp", "https://wa.me/971501234567"),
])
def test_detect_profile_valid(url, platform, canonical):
    match = detect_profile(url)
    assert match is not None and match.platform == platform and match.url == canonical


@pytest.mark.parametrize("url", [
    "https://www.instagram.com/p/abc123/",      # a post
    "https://x.com/search?q=football",          # search page
    "https://www.facebook.com/groups/12345",    # group
    "https://www.youtube.com/watch?v=abc",      # video
    "https://chat.whatsapp.com/invite",         # group invite, not a person
    "https://wa.me/123",                        # invalid number
    "javascript:alert(1)",
])
def test_detect_profile_rejects_non_profiles(url):
    assert detect_profile(url) is None


def test_post_title_is_never_used_as_a_name():
    from app.models import SearchResult
    from app.services.profile_extractor import profile_from_result
    post = SearchResult("Shout out to the Restaurant owner of Amaya, what a night",
                        "https://x.com/altodomato/status/123", "", "q")
    assert profile_from_result(post, "AE").name == "@altodomato"
    authored = SearchResult('Jane Doe on X: "Great game today"', "https://x.com/janedoe/status/1", "", "q")
    assert profile_from_result(authored, "US").name == "Jane Doe"


def test_articles_and_foreign_numbers_are_rejected():
    from app.models import SearchResult
    from app.services.profile_extractor import website_from_result
    article = SearchResult("Restaurant owners push ahead", "https://site.com/business/restaurant-owners-push-ahead-with-new-opening", "", "q")
    assert website_from_result(article, "AE") is None
    site = SearchResult("Contact Baoli Restaurant Dubai", "https://baoli.com/contact",
                        "Call +971 4 321 4567 or +1 202 456 1111", "q")
    profile = website_from_result(site, "AE")
    assert profile.name == "Baoli Restaurant Dubai"
    assert profile.phones == {"+97143214567"}  # the US number is dropped for a UAE query


def test_clean_profile_name():
    assert clean_profile_name("Jane Doe (@janedoe) • Instagram photos and videos") == "Jane Doe"
    assert clean_profile_name("Jane Doe - Founder - Acme | LinkedIn") == "Jane Doe"
    assert clean_profile_name("Home | Acme Grill") == "Acme Grill"
    assert clean_profile_name("Instagram", handle="acme") == "@acme"  # real handle, never invented


# --------------------------------------------------------------------------- #
# Query understanding
# --------------------------------------------------------------------------- #
def test_search_queries_cannot_be_hijacked():
    from app.services.query_service import _clean, build_search_queries
    u = rule_based_understanding('Cricket fans in Ahmedabad" site:evil.com')
    queries = build_search_queries(_clean(u, "q"), "site:instagram.com")
    for q in queries:
        assert q.count("site:") == 1      # only our own site filter survives
        assert q.count('"') == 2          # only our own quotes around the location


def test_rule_based_understanding():
    u = rule_based_understanding("Football lovers in Washington DC in USA")
    assert u.interest == "Football"
    assert u.audience == "Football lovers"
    assert u.location.city == "Washington DC"
    assert u.location.country_code == "US"
    assert u.target_type == "individual"


def test_llm_failure_falls_back_to_rules(monkeypatch):
    llm = get_llm_service()
    monkeypatch.setattr(llm, "_api_key", "fake-key")  # pretend the LLM is configured

    async def failing(**kwargs):
        raise LLMError("simulated outage")

    monkeypatch.setattr(llm, "structured", failing)
    understanding, source = asyncio.run(understand_query("Restaurant owners in Dubai UAE"))
    assert source == "rule_based"
    assert understanding.location.country_code == "AE"

    # The fallback answer must not be cached: once the LLM recovers, it is used again.
    async def working(**kwargs):
        return understanding.model_copy(update={"interest": "Restaurants"})

    monkeypatch.setattr(llm, "structured", working)
    _, source = asyncio.run(understand_query("Restaurant owners in Dubai UAE"))
    assert source == "llm"


def _fake_gemini(monkeypatch, status: int, payload: dict):
    """Make the Gemini provider receive a canned HTTP response."""
    import httpx

    from app.services import llm_service

    llm = get_llm_service()
    monkeypatch.setattr(llm, "provider", "gemini")
    monkeypatch.setattr(llm, "_api_key", "fake-key")

    async def fake_request(method, url, **kwargs):
        assert kwargs["headers"]["x-goog-api-key"] == "fake-key"  # key sent in a header
        assert "key=" not in url                                   # never in the URL
        return httpx.Response(status, json=payload)

    monkeypatch.setattr(llm_service, "request_with_retries", fake_request)
    return llm


def test_gemini_structured_output(monkeypatch):
    from app.services.relevance_service import RelevanceBatch
    answer = '{"verdicts": [{"id": 0, "audience_match": true, "location_match": false, "genuine_prospect": true, "reason": "fan bio"}]}'
    llm = _fake_gemini(monkeypatch, 200, {
        "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": answer}]}}],
        "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20},
    })
    result = asyncio.run(llm.structured(system="s", user="u", output_model=RelevanceBatch))
    assert result.verdicts[0].audience_match is True


@pytest.mark.parametrize("status, payload, message", [
    (200, {"candidates": []}, "no output"),
    (200, {"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": []}}]}, "cut off"),
    (200, {"candidates": [{"finishReason": "SAFETY"}]}, "declined"),
    (200, {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "not json"}]}}]}, "schema"),
    (403, {}, "invalid Gemini API key"),
    (429, {}, "rate limit"),
])
def test_gemini_errors_become_llm_errors(monkeypatch, status, payload, message):
    from app.services.relevance_service import RelevanceBatch
    llm = _fake_gemini(monkeypatch, status, payload)
    with pytest.raises(LLMError, match=message):
        asyncio.run(llm.structured(system="s", user="u", output_model=RelevanceBatch))


def test_country_name_is_mapped_to_iso_code():
    from app.schemas import Location, QueryUnderstanding
    from app.services.query_service import _clean
    u = QueryUnderstanding(is_valid_query=True, interest="Football", audience="Football lovers",
                           location=Location(city="Washington DC", country="USA", country_code="USA"),
                           target_type="individual", keywords=["football"], search_terms=["football fan"],
                           location_aliases=[])
    assert _clean(u, "q").location.country_code == "US"


# --------------------------------------------------------------------------- #
# Deduplication
# --------------------------------------------------------------------------- #
def _p(platform, name, url, handle=None, **kw):
    return RawProfile(platform=platform, name=name, url=url, handle=handle, **kw)


def test_same_name_alone_is_not_merged():
    groups = consolidate([
        _p("LinkedIn", "John Smith", "https://www.linkedin.com/in/john-smith-1", "john-smith-1"),
        _p("Instagram", "John Smith", "https://www.instagram.com/jsmith_dc/", "jsmith_dc"),
    ])
    assert len(groups) == 2


def test_merge_by_shared_email_and_cross_link():
    groups = consolidate([
        _p("Google", "Saffron Grill", "https://saffrongrill.ae",
           linked_urls={"https://www.instagram.com/saffrongrill/"}),
        _p("Instagram", "Saffron Grill", "https://www.instagram.com/saffrongrill/", "saffrongrill"),
        _p("Facebook", "Saffron Grill Dubai", "https://www.facebook.com/sgdubai", "sgdubai", emails={"hi@sg.ae"}),
        _p("Google", "Saffron Grill", "https://saffrongrill.ae", emails={"hi@sg.ae"}),  # same URL again
    ])
    assert len(groups) == 1
    assert {p.platform for p in groups[0].profiles} == {"Google", "Instagram", "Facebook"}


def test_merge_by_same_handle_and_similar_name():
    groups = consolidate([
        _p("X", "Jane Doe", "https://x.com/janedoe", "janedoe"),
        _p("Instagram", "Jane Doe", "https://www.instagram.com/jane.doe/", "jane.doe"),
        _p("Threads", "Robert Brown", "https://www.threads.com/@janedoe", "janedoe"),  # same handle, other name
    ])
    assert sorted(len(g.profiles) for g in groups) == [1, 2]


# --------------------------------------------------------------------------- #
# Relevance
# --------------------------------------------------------------------------- #
def test_llm_can_reject_non_genuine_profile():
    u = rule_based_understanding("Football lovers in Washington DC in USA")
    group = ProspectGroup([_p("Google", "Football News DC", "https://news.example", text="football news Washington DC")])
    signals = rule_signals(group, u)
    assert signals.interest_terms and signals.location_level == "local"
    verdict = CandidateVerdict(id=0, audience_match=True, location_match=True,
                               genuine_prospect=False, reason="news site")
    assert combine(group, u, signals, verdict).keep is False


def test_llm_overrules_misleading_location_match():
    u = rule_based_understanding("Startup founders in London UK")
    group = ProspectGroup([_p("LinkedIn", "Tomer London", "https://www.linkedin.com/in/tomer",
                              text="Tomer London - Co-founder of a startup, based in San Francisco")])
    signals = rule_signals(group, u)
    assert signals.location_level == "local"  # the surname matches "London"
    verdict = CandidateVerdict(id=0, audience_match=True, location_match=False,
                               genuine_prospect=True, reason="founder based in San Francisco")
    result = combine(group, u, signals, verdict)
    assert result.keep and result.status == "likely"  # still a founder, but not in London
    assert result.location_match is False and result.location_label is None
    assert not any("location" in e for e in result.evidence if not e.startswith("AI review"))


def test_all_candidates_are_reviewed_in_batches(monkeypatch):
    from app.services import relevance_service
    llm = get_llm_service()
    monkeypatch.setattr(llm, "_api_key", "fake-key")
    calls = []

    async def fake_verdicts(groups, u):
        calls.append(len(groups))
        return {i: CandidateVerdict(id=i, audience_match=True, location_match=True,
                                    genuine_prospect=True, reason="ok") for i in range(len(groups))}

    monkeypatch.setattr(relevance_service, "llm_verdicts", fake_verdicts)
    u = rule_based_understanding("Cricket lovers in Ahmedabad India")
    groups = [ProspectGroup([_p("X", f"Fan {i}", f"https://x.com/fan{i}", f"fan{i}", text="cricket")])
              for i in range(90)]
    results = asyncio.run(relevance_service.evaluate(groups, u))
    assert calls == [40, 40, 10]
    assert all(r.evaluated_by == "rules+llm" for _, r in results)


def test_profile_without_interest_evidence_is_dropped():
    u = rule_based_understanding("Football lovers in Washington DC in USA")
    group = ProspectGroup([_p("Instagram", "A Person", "https://www.instagram.com/a/", text="Coffee in Washington DC")])
    assert combine(group, u, rule_signals(group, u), None).keep is False


# --------------------------------------------------------------------------- #
# Security
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("url", [
    "http://127.0.0.1/", "http://localhost/", "http://169.254.169.254/latest/meta-data",
    "http://10.0.0.1/", "file:///etc/passwd", "https://example.com:8443/", "http://user:pw@example.com/",
])
def test_ssrf_guard_blocks_internal_urls(url):
    with pytest.raises(UnsafeURLError):
        asyncio.run(assert_public_url(url))


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("body", [{}, {"query": ""}, {"query": "  "}, {"query": "12345"},
                                  {"query": "a" * 301}, {"query": "Cricket fans", "extra": 1}])
def test_api_rejects_invalid_input(client, body):
    response = client.post("/prospect-discovery", json=body)
    assert response.status_code == 422
    assert response.json()["success"] is False


def test_api_end_to_end(client):
    response = client.post("/prospect-discovery", json={"query": "Football lovers in Washington DC in USA"})
    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert data["total_prospects"] == len(data["prospects"])
    names = [p["name"] for p in data["prospects"]]
    assert len(names) == len(set(names)), "duplicate prospects returned"

    for prospect in data["prospects"]:
        assert prospect["name"]
        assert isinstance(prospect["social_link"], list) and prospect["social_link"]
        for link in prospect["social_link"]:
            assert link["platform"] in VALID_PLATFORMS
            assert link["link"].startswith(("https://", "mailto:", "tel:"))
            assert "google.com/search" not in link["link"]

    by_name = {p["name"]: p for p in data["prospects"]}
    # Jane's Instagram + X were consolidated, and her public email is attached.
    jane = {link["link"] for link in by_name["Jane Doe"]["social_link"]}
    assert {"https://www.instagram.com/janedoe/", "https://x.com/janedoe", "mailto:jane@doe.io"} <= jane
    assert by_name["Jane Doe"]["relevance"]["status"] == "confirmed"
    assert by_name["Jane Doe"]["location"] == "Washington DC, USA"

    # Website enriched with owner-published contacts and its own Instagram + WhatsApp.
    kickers = {link["link"] for link in by_name["Capital Kickers"]["social_link"]}
    assert {"mailto:hello@capitalkickers.org", "https://www.instagram.com/capitalkickers/",
            "https://wa.me/12024561111"} <= kickers

    assert "John Smith" not in by_name                       # no football evidence
    assert not any("Top 10" in n for n in names)              # listicles are not prospects
    statuses = {s["platform"]: s["status"] for s in data["platforms"]}
    assert statuses["YouTube"] == "ok" and len(statuses) == 9


def test_api_isolates_platform_failure(client, monkeypatch):
    original = search_service.google_search

    async def flaky(query, country_code=None):
        if "site:facebook.com" in query:
            raise search_service.SearchError("simulated outage")
        return await original(query, country_code)

    monkeypatch.setattr(search_service, "google_search", flaky)
    data = client.post("/prospect-discovery", json={"query": "Football lovers in Washington DC in USA"}).json()
    assert data["success"] is True
    facebook = next(s for s in data["platforms"] if s["platform"] == "Facebook")
    assert facebook["status"] == "failed" and "simulated outage" in facebook["message"]
    assert data["total_prospects"] > 0


def test_api_all_sources_failed(client, monkeypatch):
    async def down(query, country_code=None):
        raise search_service.SearchError("down")

    monkeypatch.setattr(search_service, "google_search", down)
    response = client.post("/prospect-discovery", json={"query": "Startup founders in London UK"})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "all_sources_failed"
