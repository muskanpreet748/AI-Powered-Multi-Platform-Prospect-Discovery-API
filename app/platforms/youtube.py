"""YouTube - public creator/channel profiles via the official YouTube Data API v3.

1. search.list (type=channel)  -> channel IDs matching the audience + location
2. channels.list               -> title, description, handle and country of each channel

Contacts/social links are taken only from the channel's public description.
If no YOUTUBE_API_KEY is configured, it falls back to Google's index of youtube.com.
"""

from app.config import get_settings
from app.models import RawProfile
from app.platforms.base import DiscoveryContext, PlatformError, SiteSearchPlatform
from app.services.profile_extractor import add_contacts_from_text
from app.services.query_service import location_phrase
from app.utils.http_client import ExternalAPIError, request_with_retries

YT_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YT_CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"


class YouTubePlatform(SiteSearchPlatform):
    name = "YouTube"
    site_filter = "site:youtube.com"

    def availability(self, settings) -> str | None:
        if settings.youtube_api_enabled or settings.search_enabled:
            return None
        return "YOUTUBE_API_KEY and SERPER_API_KEY not configured"

    async def discover(self, ctx: DiscoveryContext) -> list[RawProfile]:
        if not get_settings().youtube_api_enabled:
            return await super().discover(ctx)  # Google-index fallback
        return await self._discover_with_api(ctx)

    async def _discover_with_api(self, ctx: DiscoveryContext) -> list[RawProfile]:
        settings = get_settings()
        u = ctx.understanding
        location = location_phrase(u) or ""

        channel_ids: list[str] = []
        for term in u.search_terms[: settings.max_queries_per_platform]:
            params = {
                "part": "snippet", "type": "channel",
                "q": f"{term} {location}".strip(),
                "maxResults": settings.max_results_per_query,
            }
            if ctx.region:
                params["regionCode"] = ctx.region
            data = await self._get(YT_SEARCH_URL, params)
            for item in data.get("items", []):
                channel_id = (item.get("id") or {}).get("channelId")
                if channel_id and channel_id not in channel_ids:
                    channel_ids.append(channel_id)

        if not channel_ids:
            return []

        data = await self._get(YT_CHANNELS_URL, {"part": "snippet", "id": ",".join(channel_ids[:50])})
        profiles = []
        for item in data.get("items", []):
            snippet = item.get("snippet") or {}
            title = str(snippet.get("title", "")).strip()
            if not title:
                continue
            custom_url = str(snippet.get("customUrl") or "")
            if custom_url.startswith("@"):
                handle = custom_url[1:].lower()
                url = f"https://www.youtube.com/@{handle}"
            else:
                handle = None
                url = f"https://www.youtube.com/channel/{item.get('id')}"
            description = str(snippet.get("description", ""))
            profile = RawProfile(
                platform="YouTube", name=title, url=url, handle=handle,
                text=f"{title}. {description[:1000]}",
                source=f"YouTube Data API v3 ({url})",
                country_code=snippet.get("country"),
            )
            add_contacts_from_text(profile, description, ctx.region)
            profiles.append(profile)
        return profiles

    async def _get(self, url: str, params: dict) -> dict:
        # The key goes in a header (not the URL) so it never appears in logs.
        headers = {"X-Goog-Api-Key": get_settings().youtube_api_key.get_secret_value()}
        try:
            response = await request_with_retries("GET", url, params=params, headers=headers)
        except ExternalAPIError as exc:
            raise PlatformError(f"YouTube API unreachable ({exc})") from exc
        if response.status_code == 403:
            raise PlatformError("YouTube API quota exceeded or key not permitted")
        if response.status_code != 200:
            raise PlatformError(f"YouTube API returned HTTP {response.status_code}")
        try:
            return response.json()
        except ValueError as exc:
            raise PlatformError("YouTube API returned invalid JSON") from exc
