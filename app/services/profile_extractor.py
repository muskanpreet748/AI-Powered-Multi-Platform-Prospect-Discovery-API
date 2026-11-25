"""
Profile extraction - turns raw public data into `RawProfile` objects.

Sources of data:
  * search results (title + snippet + URL)
  * public websites of prospects (contact links, social links in the HTML)

Everything extracted here is copied from public content and validated.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from app.models import RawProfile, SearchResult
from app.utils.http_client import fetch_public_html
from app.utils.normalizers import (
    clean_profile_name, detect_profile, domain_of, website_url, whatsapp_link,
)
from app.utils.validators import (
    extract_emails, extract_phones, extract_whatsapp_numbers, validate_email, validate_phone,
)

logger = logging.getLogger(__name__)

# Profile URLs written inside text (bios/snippets), with or without "https://".
_TEXT_PROFILE_URL = re.compile(
    r"(?:https?://)?(?:www\.)?(?:instagram\.com|facebook\.com|x\.com|twitter\.com|linkedin\.com|"
    r"youtube\.com|threads\.(?:net|com)|wa\.me)/[^\s\"'<>),]+",
    re.I,
)

# Websites that are directories, marketplaces, media or social networks -
# not the prospect's own site.
_NON_PROSPECT_DOMAINS = (
    "linkedin.com", "facebook.com", "instagram.com", "x.com", "twitter.com", "youtube.com",
    "threads.net", "threads.com", "tiktok.com", "pinterest.com", "reddit.com", "quora.com",
    "wikipedia.org", "yelp.com", "tripadvisor.", "zomato.com", "justdial.com", "indeed.com",
    "glassdoor.", "clutch.co", "crunchbase.com", "eventbrite.", "meetup.com", "medium.com",
    "google.", "amazon.", "apple.com", "yellowpages.", "booking.com", "timeout.com",
)
# Hosts where many different people have pages under one domain.
_MULTI_TENANT_HOSTS = {"linktr.ee", "sites.google.com", "about.me", "beacons.ai", "bio.link", "carrd.co"}

_LISTING_TITLE = re.compile(
    r"\b(?:top|best)\s+\d+|\b\d+\s+(?:best|top)\b|\blist of\b|\bdirectory\b|\branking|"
    r"\breviews?\b|\bnear me\b|\bjobs?\b|\bnews\b|\bthings to do\b|\bguide to\b",
    re.I,
)


def is_prospect_website(url: str, title: str) -> bool:
    domain = domain_of(url)
    if any(domain == d or domain.endswith("." + d) or (d.endswith(".") and d in domain)
           for d in _NON_PROSPECT_DOMAINS):
        return False
    return not _LISTING_TITLE.search(title)


def add_contacts_from_text(profile: RawProfile, text: str, region: str | None) -> None:
    """Copy emails, phones, WhatsApp numbers and profile links that appear in public text."""
    profile.emails |= extract_emails(text)
    profile.phones |= extract_phones(text, region, limit=2)
    profile.whatsapp |= {whatsapp_link(p) for p in extract_whatsapp_numbers(text, region)}
    for raw_url in _TEXT_PROFILE_URL.findall(text or ""):
        url = raw_url if raw_url.lower().startswith("http") else "https://" + raw_url
        match = detect_profile(url.rstrip("."), region)
        if not match:
            continue
        if match.platform == "WhatsApp":
            profile.whatsapp.add(match.url)
        elif match.url != profile.url:
            profile.linked_urls.add(match.url)


def profile_from_result(result: SearchResult, region: str | None) -> RawProfile | None:
    """A search result that points to a social profile -> RawProfile (else None)."""
    match = detect_profile(result.link, region)
    if not match or match.platform == "WhatsApp":
        return None
    name = clean_profile_name(result.title, match.handle)
    if not name:
        return None
    profile = RawProfile(
        platform=match.platform,
        name=name,
        url=match.url,
        handle=match.handle,
        text=f"{result.title}. {result.snippet}",
        source=result.link,
    )
    add_contacts_from_text(profile, result.snippet, region)
    return profile


def website_from_result(result: SearchResult, region: str | None) -> RawProfile | None:
    """A search result that is a prospect's own website -> RawProfile (platform "Google")."""
    url = website_url(result.link)
    if not url or not is_prospect_website(url, result.title):
        return None
    # One website = one entity: "/join" and "/about" pages map to the site root,
    # except on shared hosts where each path belongs to a different person.
    parts = urlsplit(url)
    if domain_of(url) not in _MULTI_TENANT_HOSTS:
        url = f"{parts.scheme}://{parts.hostname}"
    name = clean_profile_name(result.title)
    if not name:
        return None
    profile = RawProfile(
        platform="Google", name=name, url=url,
        text=f"{result.title}. {result.snippet}", source=result.link,
    )
    add_contacts_from_text(profile, result.snippet, region)
    return profile


# --------------------------------------------------------------------------- #
# Website enrichment
# --------------------------------------------------------------------------- #
@dataclass
class PageData:
    emails: set[str] = field(default_factory=set)
    phones: set[str] = field(default_factory=set)
    whatsapp: set[str] = field(default_factory=set)
    social: dict[str, set[str]] = field(default_factory=dict)  # platform -> profile URLs
    site_name: str | None = None
    description: str = ""
    contact_page: str | None = None


def parse_page(html: str, page_url: str, region: str | None) -> PageData:
    data = PageData()
    soup = BeautifulSoup(html, "html.parser")
    page_host = urlsplit(page_url).hostname

    og_name = soup.find("meta", property="og:site_name")
    if og_name and og_name.get("content"):
        data.site_name = og_name["content"].strip()[:80]
    description = soup.find("meta", attrs={"name": "description"})
    if description and description.get("content"):
        data.description = description["content"].strip()[:500]

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        lower = href.lower()
        if lower.startswith("mailto:"):
            email = validate_email(href)
            if email:
                data.emails.add(email)
        elif lower.startswith("tel:"):
            phone = validate_phone(href[4:], region)
            if phone:
                data.phones.add(phone)
        else:
            absolute = urljoin(page_url, href)
            match = detect_profile(absolute, region)
            if match and match.platform == "WhatsApp":
                data.whatsapp.add(match.url)
            elif match:
                data.social.setdefault(match.platform, set()).add(match.url)
            elif (urlsplit(absolute).hostname == page_host and "contact" in lower
                  and data.contact_page is None):
                data.contact_page = absolute

    text = soup.get_text(" ", strip=True)[:200_000]
    data.emails |= extract_emails(text)
    data.whatsapp |= {whatsapp_link(p) for p in extract_whatsapp_numbers(text, region)}
    return data


def _merge_page_into_profile(profile: RawProfile, page: PageData) -> None:
    # A page listing many different emails is probably a directory: we cannot
    # tell which one belongs to this prospect, so we keep none of them.
    if len(page.emails) <= 3:
        profile.emails |= page.emails
    if len(page.phones) <= 3:
        profile.phones |= page.phones
    profile.whatsapp |= page.whatsapp
    # Exactly one link per platform = the site's own account. Several different
    # links (e.g. partners, the web designer) are ambiguous, so we skip them.
    for urls in page.social.values():
        if len(urls) == 1:
            profile.linked_urls |= urls
    if page.description:
        profile.text += f" {page.description}"


async def enrich_website(profile: RawProfile, region: str | None) -> None:
    """Fetch the prospect's public website (and contact page) to collect published contacts."""
    fetched = await fetch_public_html(profile.url)
    if not fetched:
        return
    final_url, html = fetched
    page = parse_page(html, final_url, region)
    site_name = clean_profile_name(page.site_name or "")
    if site_name:
        profile.name = site_name
    _merge_page_into_profile(profile, page)

    if not profile.emails and page.contact_page:
        contact = await fetch_public_html(page.contact_page)
        if contact:
            _merge_page_into_profile(profile, parse_page(contact[1], contact[0], region))


async def enrich_websites(profiles: list[RawProfile], region: str | None, limit: int) -> None:
    targets = profiles[:limit]
    results = await asyncio.gather(*(enrich_website(p, region) for p in targets), return_exceptions=True)
    for profile, result in zip(targets, results):
        if isinstance(result, Exception):
            logger.info("Enrichment failed for %s: %s", domain_of(profile.url), type(result).__name__)
