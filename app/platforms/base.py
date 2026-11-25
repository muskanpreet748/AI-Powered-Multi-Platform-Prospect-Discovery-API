"""
Base classes for platform integrations.

Every platform is an independent module that implements the same interface:

    availability(settings) -> None if it can run, else a reason string
    discover(ctx)          -> list[RawProfile]

The orchestrator (`prospect_service`) runs all platforms in parallel and
isolates their failures, so one broken platform never breaks the request.
To add a new platform: subclass `BasePlatform` (or `SiteSearchPlatform`) and
register it in `app/platforms/__init__.py`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.config import Settings
from app.models import RawProfile
from app.schemas import QueryUnderstanding
from app.services.profile_extractor import profile_from_result, website_from_result
from app.services.query_service import build_search_queries
from app.services.search_service import search_many


@dataclass
class DiscoveryContext:
    understanding: QueryUnderstanding

    @property
    def region(self) -> str | None:
        """ISO country code used for localised search and phone validation."""
        return self.understanding.location.country_code


class PlatformError(Exception):
    """A platform could not complete discovery (reported in `platforms` status)."""


class BasePlatform(ABC):
    name: str = ""

    def availability(self, settings: Settings) -> str | None:
        return None

    @abstractmethod
    async def discover(self, ctx: DiscoveryContext) -> list[RawProfile]:
        ...


class SiteSearchPlatform(BasePlatform):
    """
    For platforms that offer no public people-search API (LinkedIn, Instagram,
    Facebook, X, Threads). We use Google search restricted to the platform's
    domain (`site:` operator) - only publicly indexed profiles are found and we
    never log in to or scrape the platform itself.
    """

    site_filter: str = ""

    def availability(self, settings: Settings) -> str | None:
        return None if settings.search_enabled else "SERPER_API_KEY not configured"

    def site_filter_for(self, ctx: DiscoveryContext) -> str:
        return self.site_filter

    async def discover(self, ctx: DiscoveryContext) -> list[RawProfile]:
        queries = build_search_queries(ctx.understanding, self.site_filter_for(ctx))
        results = await search_many(queries, ctx.region)
        profiles = []
        for result in results:
            profile = profile_from_result(result, ctx.region)
            if profile and profile.platform == self.name:
                profile.source = result.link
                profiles.append(profile)
        return profiles


class ContactSearchPlatform(BasePlatform):
    """
    For contact channels (Email, WhatsApp): search the public web for pages
    where prospects *publish* that contact, and keep only results that really
    contain one. The profile keeps its real platform (e.g. Instagram/website).
    """

    extra_terms: str = ""

    def availability(self, settings: Settings) -> str | None:
        return None if settings.search_enabled else "SERPER_API_KEY not configured"

    def has_contact(self, profile: RawProfile) -> bool:
        raise NotImplementedError

    async def discover(self, ctx: DiscoveryContext) -> list[RawProfile]:
        queries = build_search_queries(ctx.understanding, extra=self.extra_terms)
        results = await search_many(queries, ctx.region)
        profiles = []
        for result in results:
            profile = profile_from_result(result, ctx.region) or website_from_result(result, ctx.region)
            if profile and self.has_contact(profile):
                profile.source = result.link
                profiles.append(profile)
        return profiles
