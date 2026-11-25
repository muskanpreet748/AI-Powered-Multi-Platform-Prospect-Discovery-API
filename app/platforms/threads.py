"""Threads - public profiles (threads.com/@handle).

The Threads API only works for the authenticated user's own account, so
discovery uses Google's public index of threads.net / threads.com.
"""

from app.platforms.base import SiteSearchPlatform


class ThreadsPlatform(SiteSearchPlatform):
    name = "Threads"
    site_filter = "(site:threads.net OR site:threads.com)"
