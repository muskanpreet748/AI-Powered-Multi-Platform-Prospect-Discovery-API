"""Google - publicly indexed websites of prospects (individuals and businesses).

Google is a *discovery source*: we never return a search-result URL as a
prospect. Listicles/directories/news are filtered out, and each remaining
website is visited (safely, respecting robots.txt) to collect the contact
details and social accounts the owner published on it.
"""

from app.config import get_settings
from app.models import RawProfile
from app.platforms.base import BasePlatform, DiscoveryContext
from app.services.profile_extractor import enrich_websites, website_from_result
from app.services.query_service import build_search_queries
from app.services.search_service import search_many

# Social networks are covered by their own platform modules; directories are noise.
_EXCLUDED_SITES = (
    "linkedin.com", "facebook.com", "instagram.com", "x.com", "twitter.com",
    "youtube.com", "threads.net", "yelp.com", "tripadvisor.com", "wikipedia.org",
)
_EXCLUDE_FILTER = " ".join(f"-site:{site}" for site in _EXCLUDED_SITES)


class GooglePlatform(BasePlatform):
    name = "Google"

    def availability(self, settings) -> str | None:
        return None if settings.search_enabled else "SERPER_API_KEY not configured"

    async def discover(self, ctx: DiscoveryContext) -> list[RawProfile]:
        queries = build_search_queries(ctx.understanding, extra=_EXCLUDE_FILTER)
        results = await search_many(queries, ctx.region)

        websites: list[RawProfile] = []
        seen: set[str] = set()
        for result in results:
            profile = website_from_result(result, ctx.region)
            if profile and profile.url not in seen:
                seen.add(profile.url)
                websites.append(profile)

        await enrich_websites(websites, ctx.region, get_settings().max_websites_to_enrich)
        return websites
