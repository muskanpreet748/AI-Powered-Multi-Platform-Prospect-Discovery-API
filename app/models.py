"""
Internal data structures used while a request is being processed.

These are plain dataclasses (not part of the public API). A `RawProfile` is
one profile found on one platform. Several RawProfiles that belong to the same
person are later merged into one `ProspectGroup`.
"""

from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """One organic result returned by the search API."""
    title: str
    link: str
    snippet: str
    query: str  # the search query that produced it (kept for traceability)


@dataclass
class RawProfile:
    platform: str                 # "LinkedIn", "Instagram", "Google" (website), ...
    name: str                     # display name taken from public data
    url: str                      # canonical public profile URL
    handle: str | None = None     # username, e.g. "janedoe" (if the platform has one)
    text: str = ""                # public title + snippet/bio, used as relevance evidence
    source: str = ""              # where it was found, e.g. a search query or "YouTube Data API"
    emails: set[str] = field(default_factory=set)
    phones: set[str] = field(default_factory=set)       # E.164 format, e.g. +12025550123
    whatsapp: set[str] = field(default_factory=set)     # https://wa.me/<digits>
    linked_urls: set[str] = field(default_factory=set)  # other canonical profile URLs it links to
    country_code: str | None = None                     # e.g. from YouTube channel settings


@dataclass
class ProspectGroup:
    """All profiles that were identified as the same person/business."""
    profiles: list[RawProfile]

    @property
    def all_text(self) -> str:
        return " \n".join(p.text for p in self.profiles if p.text)
