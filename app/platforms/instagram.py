"""Instagram - public profiles.

The Instagram Graph API only exposes accounts you manage (or business accounts
by exact username), so discovery uses Google's public index of instagram.com.
"""

from app.platforms.base import SiteSearchPlatform


class InstagramPlatform(SiteSearchPlatform):
    name = "Instagram"
    site_filter = "site:instagram.com"
