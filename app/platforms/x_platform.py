"""X (Twitter) - public user profiles.

X's user-search endpoints need a paid API tier, so discovery uses Google's
public index of x.com / twitter.com. Post URLs are mapped to their author's
profile (x.com/<handle>/status/<id> -> x.com/<handle>).
"""

from app.platforms.base import SiteSearchPlatform


class XPlatform(SiteSearchPlatform):
    name = "X"
    site_filter = "(site:x.com OR site:twitter.com)"
