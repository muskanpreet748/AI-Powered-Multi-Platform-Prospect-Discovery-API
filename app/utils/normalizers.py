"""
Normalizers: turn messy public data into clean, comparable values.

The most important function is `detect_profile()`. Given any URL it answers:
"Is this a public *profile* on a supported platform? If yes, what is its
canonical URL and username?" Posts, search pages, login pages, groups etc. are
rejected, so only real profile links reach the final response.
"""

import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import parse_qs, urlsplit

from app.utils.validators import validate_phone


@dataclass(frozen=True)
class ProfileMatch:
    platform: str
    url: str            # canonical URL
    handle: str | None  # username / slug


# Path segments that are platform features, not usernames.
_INSTAGRAM_RESERVED = {
    "p", "reel", "reels", "explore", "stories", "tv", "accounts", "about",
    "developer", "legal", "direct", "web", "challenge", "popular",
}
_X_RESERVED = {
    "home", "explore", "search", "i", "intent", "share", "hashtag", "settings",
    "login", "signup", "tos", "privacy", "notifications", "messages", "compose",
}
_FACEBOOK_RESERVED = {
    "groups", "events", "watch", "marketplace", "photo", "photo.php", "photos",
    "story.php", "sharer", "sharer.php", "login", "login.php", "help", "hashtag",
    "search", "reel", "reels", "share", "plugins", "policies", "privacy", "dialog",
    "permalink.php", "media", "gaming", "fundraisers", "business", "ads", "l.php",
}
_THREADS_HOSTS = {"threads.net", "threads.com"}
_X_HOSTS = {"x.com", "twitter.com"}

_INSTAGRAM_HANDLE = re.compile(r"^[A-Za-z0-9._]{1,30}$")
_X_HANDLE = re.compile(r"^[A-Za-z0-9_]{1,15}$")
_SLUG = re.compile(r"^[A-Za-z0-9._%-]{2,100}$")


def _host(netloc: str) -> str:
    host = netloc.lower().split("@")[-1].split(":")[0]
    for prefix in ("www.", "m.", "mobile.", "web.", "business."):
        if host.startswith(prefix):
            host = host[len(prefix):]
    return host


def _segments(path: str) -> list[str]:
    return [s for s in path.split("/") if s]


def detect_profile(url: str, region: str | None = None) -> ProfileMatch | None:
    """Return the canonical profile for a URL, or None if it is not a profile."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return None
    if parts.scheme not in ("http", "https"):
        return None

    host = _host(parts.netloc)
    seg = _segments(parts.path)

    # LinkedIn: /in/<slug> (people) and /company/<slug> (businesses)
    if host == "linkedin.com" or host.endswith(".linkedin.com"):
        if len(seg) >= 2 and seg[0] in ("in", "company") and _SLUG.match(seg[1]):
            slug = seg[1].lower()
            return ProfileMatch("LinkedIn", f"https://www.linkedin.com/{seg[0]}/{slug}", slug)
        return None

    # Instagram: /<handle>
    if host in ("instagram.com", "instagr.am"):
        if seg and seg[0].lower() not in _INSTAGRAM_RESERVED and _INSTAGRAM_HANDLE.match(seg[0]):
            if len(seg) == 1 or seg[1] in ("reels", "tagged", "guides"):
                handle = seg[0].lower()
                return ProfileMatch("Instagram", f"https://www.instagram.com/{handle}/", handle)
        return None

    # X / Twitter: /<handle> - a post URL (/<handle>/status/<id>) still names the profile
    if host in _X_HOSTS:
        if seg and seg[0].lower() not in _X_RESERVED and _X_HANDLE.match(seg[0]):
            if len(seg) == 1 or seg[1] in ("status", "with_replies", "media"):
                handle = seg[0].lower()
                return ProfileMatch("X", f"https://x.com/{handle}", handle)
        return None

    # Threads: /@<handle>
    if host in _THREADS_HOSTS:
        if seg and seg[0].startswith("@") and _INSTAGRAM_HANDLE.match(seg[0][1:]):
            handle = seg[0][1:].lower()
            return ProfileMatch("Threads", f"https://www.threads.com/@{handle}", handle)
        return None

    # YouTube: /@handle, /channel/<id>, /c/<name>, /user/<name>
    if host in ("youtube.com", "youtu.be") or host.endswith(".youtube.com"):
        if seg and seg[0].startswith("@") and len(seg[0]) > 1:
            handle = seg[0][1:].lower()
            return ProfileMatch("YouTube", f"https://www.youtube.com/@{handle}", handle)
        if len(seg) >= 2 and seg[0] == "channel" and re.match(r"^UC[\w-]{22}$", seg[1]):
            return ProfileMatch("YouTube", f"https://www.youtube.com/channel/{seg[1]}", None)
        if len(seg) >= 2 and seg[0] in ("c", "user") and _SLUG.match(seg[1]):
            return ProfileMatch("YouTube", f"https://www.youtube.com/{seg[0]}/{seg[1]}", seg[1].lower())
        return None

    # Facebook: /<page>, /profile.php?id=<n>, /people/<name>/<id>, /pages/<name>/<id>
    if host in ("facebook.com", "fb.com"):
        if not seg:
            return None
        first = seg[0].lower()
        if first == "profile.php":
            ids = parse_qs(parts.query).get("id", [])
            if ids and ids[0].isdigit():
                return ProfileMatch("Facebook", f"https://www.facebook.com/profile.php?id={ids[0]}", ids[0])
            return None
        if first in ("people", "pages") and len(seg) >= 3 and seg[2].isdigit():
            return ProfileMatch("Facebook", f"https://www.facebook.com/{first}/{seg[1]}/{seg[2]}", seg[2])
        if first not in _FACEBOOK_RESERVED and _SLUG.match(seg[0]) and not first.endswith(".php"):
            return ProfileMatch("Facebook", f"https://www.facebook.com/{first}", first)
        return None

    # WhatsApp click-to-chat links: wa.me/<number>, api.whatsapp.com/send?phone=<number>
    if host == "wa.me" or host.endswith("whatsapp.com"):
        number = None
        if host == "wa.me" and seg:
            number = seg[0]
        elif seg and seg[0] == "send":
            number = (parse_qs(parts.query).get("phone") or [None])[0]
        if number:
            e164 = validate_phone("+" + re.sub(r"\D", "", number), region)
            if e164:
                return ProfileMatch("WhatsApp", whatsapp_link(e164), e164)
        return None  # group invites (chat.whatsapp.com) are not personal contacts

    return None


def whatsapp_link(e164_phone: str) -> str:
    """Official click-to-chat format: https://wa.me/<digits without +>."""
    return "https://wa.me/" + e164_phone.lstrip("+")


def website_url(url: str) -> str | None:
    """Canonical form of an ordinary website URL (scheme + host + path)."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    path = parts.path.rstrip("/")
    return f"{parts.scheme}://{parts.hostname.lower()}{path}"


def domain_of(url: str) -> str:
    return _host(urlsplit(url).netloc)


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #
_TITLE_SPLIT = re.compile(r"\s+(?:\||-|–|—|•|·|/|:)\s+|\s+\(@|\s+on\s+(?:X|Twitter|Instagram|Threads)\b")
_GENERIC_NAMES = {
    "home", "homepage", "welcome", "login", "log in", "sign up", "instagram", "facebook",
    "linkedin", "x", "twitter", "threads", "youtube", "official site", "official website",
    "contact", "contact us", "about", "about us", "page not found", "untitled",
}


def clean_profile_name(title: str, handle: str | None = None) -> str | None:
    """
    Extract a display name from a search-result title such as
    "Jane Doe (@janedoe) • Instagram photos and videos" -> "Jane Doe".
    Falls back to "@handle" (real public data) - never invents a name.
    """
    for part in _TITLE_SPLIT.split(title or ""):
        name = part.strip(" \"'@|-–•")
        if 2 <= len(name) <= 80 and name.lower() not in _GENERIC_NAMES and not name.lower().startswith("http"):
            return name
    return f"@{handle}" if handle else None


def normalize_name(name: str) -> str:
    """Lowercase, remove accents and punctuation: 'José  O'Neil' -> 'jose oneil'."""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9 ]", "", text.lower())
    return " ".join(text.split())


def normalize_handle(handle: str) -> str:
    """Compare handles ignoring separators: 'jane.doe_' == 'janedoe'."""
    return re.sub(r"[^a-z0-9]", "", handle.lower())


def sanitize_search_term(term: str, max_length: int = 60) -> str:
    """Keep search terms plain: strip quotes and search operators such as 'site:'."""
    term = re.sub(r"\b\w+:", " ", term)              # remove operators like site:, inurl:
    term = re.sub(r"[^\w\s'&.-]", " ", term)
    return " ".join(term.split())[:max_length]
