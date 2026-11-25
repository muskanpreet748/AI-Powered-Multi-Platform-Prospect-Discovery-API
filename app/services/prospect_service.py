"""
Prospect service - the orchestrator that runs the whole pipeline:

  query -> understanding (LLM) -> parallel platform discovery
        -> consolidation (dedupe) -> relevance filtering -> JSON response

Each platform runs concurrently with its own timeout. A failing platform is
reported in `platforms[]` but never breaks the request.
"""

import asyncio
import logging
import time

from app.config import get_settings
from app.models import ProspectGroup, RawProfile
from app.platforms import get_platforms
from app.platforms.base import BasePlatform, DiscoveryContext, PlatformError
from app.schemas import PlatformStatus, Prospect, ProspectResponse, Relevance, SocialLink
from app.services.deduplication_service import consolidate
from app.services.query_service import understand_query
from app.services.relevance_service import RelevanceResult, evaluate
from app.services.search_service import SearchError
from app.utils.normalizers import detect_profile

logger = logging.getLogger(__name__)

# Order of links inside `social_link`, and preference when choosing a display name.
_PROFILE_ORDER = ["LinkedIn", "Facebook", "Instagram", "X", "YouTube", "Threads", "Google"]


class DiscoveryError(Exception):
    """A request-level failure that becomes a JSON error response."""

    def __init__(self, code: str, message: str, status_code: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


async def _run_platform(platform: BasePlatform, ctx: DiscoveryContext, timeout: float
                        ) -> tuple[PlatformStatus, list[RawProfile]]:
    started = time.perf_counter()
    try:
        profiles = await asyncio.wait_for(platform.discover(ctx), timeout)
    except asyncio.TimeoutError:
        return PlatformStatus(platform=platform.name, status="failed", message=f"timed out after {timeout:.0f}s"), []
    except (PlatformError, SearchError) as exc:
        logger.warning("%s discovery failed: %s", platform.name, exc)
        return PlatformStatus(platform=platform.name, status="failed", message=str(exc)), []
    except Exception:  # a bug in one platform must not break the others
        logger.exception("Unexpected error in %s discovery", platform.name)
        return PlatformStatus(platform=platform.name, status="failed", message="unexpected error"), []

    logger.info("%s: %d profiles in %.1fs", platform.name, len(profiles), time.perf_counter() - started)
    return PlatformStatus(platform=platform.name, status="ok", profiles_found=len(profiles)), profiles


def _order(platform: str) -> int:
    return _PROFILE_ORDER.index(platform) if platform in _PROFILE_ORDER else len(_PROFILE_ORDER)


def _choose_name(group: ProspectGroup) -> str:
    profiles = sorted(group.profiles, key=lambda p: _order(p.platform))
    for profile in profiles:
        if not profile.name.startswith("@"):
            return profile.name
    return profiles[0].name


def build_prospect(group: ProspectGroup, relevance: RelevanceResult) -> Prospect:
    links: dict[str, SocialLink] = {}

    def add(platform: str, link: str) -> None:
        if link not in links:
            links[link] = SocialLink(platform=platform, link=link)

    profiles = sorted(group.profiles, key=lambda p: _order(p.platform))
    for profile in profiles:
        add(profile.platform, profile.url)
    # Profiles the prospect links to from their own website/bio (owner-published).
    for profile in profiles:
        for url in sorted(profile.linked_urls):
            match = detect_profile(url)
            if match and match.platform != "WhatsApp":
                add(match.platform, match.url)
    for email in sorted({e for p in profiles for e in p.emails}):
        add("Email", f"mailto:{email}")
    for phone in sorted({ph for p in profiles for ph in p.phones}):
        add("Phone", f"tel:{phone}")
    for whatsapp in sorted({w for p in profiles for w in p.whatsapp}):
        add("WhatsApp", whatsapp)

    return Prospect(
        name=_choose_name(group),
        location=relevance.location_label,
        social_link=list(links.values()),
        relevance=Relevance(
            score=relevance.score,
            status=relevance.status,
            interest_match=relevance.interest_match,
            location_match=relevance.location_match,
            evidence=relevance.evidence,
            evaluated_by=relevance.evaluated_by,
        ),
        sources=sorted({p.source for p in profiles if p.source})[:10],
    )


async def discover_prospects(query: str) -> ProspectResponse:
    settings = get_settings()
    started = time.perf_counter()

    # 1-4. Understand the query
    understanding, source = await understand_query(query)
    if not understanding.is_valid_query:
        raise DiscoveryError(
            "invalid_query",
            "The query does not describe people or businesses to find. "
            "Example: 'Football lovers in Washington DC in USA'.",
            422,
        )

    # 5-7. Discover on every available platform in parallel
    ctx = DiscoveryContext(understanding=understanding)
    statuses: list[PlatformStatus] = []
    runnable: list[BasePlatform] = []
    for platform in get_platforms():
        reason = platform.availability(settings)
        if reason:
            statuses.append(PlatformStatus(platform=platform.name, status="skipped", message=reason))
        else:
            runnable.append(platform)

    if not runnable:
        raise DiscoveryError("no_sources_configured",
                             "No discovery source is configured. Set SERPER_API_KEY (see .env.example).", 503)

    outcomes = await asyncio.gather(
        *(_run_platform(p, ctx, settings.platform_timeout_seconds) for p in runnable)
    )
    profiles: list[RawProfile] = []
    for status, found in outcomes:
        statuses.append(status)
        profiles.extend(found)

    if all(status.status == "failed" for status, _ in outcomes):
        raise DiscoveryError("all_sources_failed",
                             "All discovery sources failed. Please try again later.", 502)

    # 8-10. Consolidate duplicates, then keep only relevant prospects
    groups = consolidate(profiles)
    scored = await evaluate(groups, understanding)
    prospects = [build_prospect(group, rel) for group, rel in scored[: settings.max_prospects]]

    platform_order = [p.name for p in get_platforms()]
    statuses.sort(key=lambda s: platform_order.index(s.platform))
    logger.info("Query done in %.1fs: %d raw profiles -> %d groups -> %d prospects",
                time.perf_counter() - started, len(profiles), len(groups), len(prospects))

    # 11-12. Build the response (validated by Pydantic)
    return ProspectResponse(
        query=query,
        total_prospects=len(prospects),
        prospects=prospects,
        understanding=understanding,
        understanding_source=source,
        platforms=statuses,
    )
