"""Facebook - public profiles and pages.

Facebook's Pages Search API requires app review and does not search people,
so discovery uses Google's public index of facebook.com.
"""

from app.platforms.base import SiteSearchPlatform


class FacebookPlatform(SiteSearchPlatform):
    name = "Facebook"
    site_filter = "site:facebook.com"
