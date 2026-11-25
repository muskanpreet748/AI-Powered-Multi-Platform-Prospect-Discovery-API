"""LinkedIn - public professional profiles (/in/) and company pages (/company/).

LinkedIn's official APIs do not allow searching members, and scraping is
forbidden by its terms, so profiles are discovered from Google's public index.
"""

from app.platforms.base import DiscoveryContext, SiteSearchPlatform


class LinkedInPlatform(SiteSearchPlatform):
    name = "LinkedIn"

    def site_filter_for(self, ctx: DiscoveryContext) -> str:
        if ctx.understanding.target_type == "business":
            return "(site:linkedin.com/in OR site:linkedin.com/company)"
        return "site:linkedin.com/in"
