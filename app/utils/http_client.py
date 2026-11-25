"""
Shared HTTP client + safety helpers.

1. `request_with_retries()` - calls third-party APIs (Serper, YouTube) and retries
   on rate limits (429) and temporary server errors with exponential backoff.

2. `fetch_public_html()` - downloads a public web page found in search results.
   Those URLs come from the internet, so they are untrusted. To prevent SSRF
   (Server-Side Request Forgery - tricking our server into calling internal
   addresses like 127.0.0.1 or cloud metadata at 169.254.169.254) we:
     * allow only http/https on default ports,
     * resolve the hostname and reject private/loopback/link-local IPs,
     * follow redirects manually and re-check every hop,
     * cap the download size and accept only HTML,
     * respect the site's robots.txt.
"""

import asyncio
import ipaddress
import logging
import socket
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

USER_AGENT = "ProspectDiscoveryBot/1.0 (public-data research; respects robots.txt)"
MAX_REDIRECTS = 3
MAX_HTML_BYTES = 1_500_000
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """One shared connection pool for the whole app (faster than a client per call)."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=get_settings().http_timeout_seconds,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=False,
        )
    return _client


async def close_http_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


# --------------------------------------------------------------------------- #
# Calling third-party APIs
# --------------------------------------------------------------------------- #
class ExternalAPIError(Exception):
    """A third-party API failed in a way that retrying will not fix."""


async def request_with_retries(method: str, url: str, *, attempts: int = 3, **kwargs) -> httpx.Response:
    client = get_http_client()
    delay = 1.0
    for attempt in range(1, attempts + 1):
        try:
            response = await client.request(method, url, **kwargs)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if attempt == attempts:
                raise ExternalAPIError(f"network error: {type(exc).__name__}") from exc
        else:
            if response.status_code not in RETRYABLE_STATUS:
                return response
            if attempt == attempts:
                return response
            retry_after = response.headers.get("retry-after", "")
            if retry_after.isdigit():
                delay = min(float(retry_after), 10.0)
            logger.warning("HTTP %s from %s, retry %d/%d in %.1fs",
                           response.status_code, urlsplit(url).netloc, attempt, attempts - 1, delay)
        await asyncio.sleep(delay)
        delay = min(delay * 2, 10.0)
    raise ExternalAPIError("unreachable")  # pragma: no cover


# --------------------------------------------------------------------------- #
# Fetching untrusted public web pages safely
# --------------------------------------------------------------------------- #
class UnsafeURLError(ValueError):
    pass


async def assert_public_url(url: str) -> None:
    """Raise UnsafeURLError unless the URL points to a public internet host."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise UnsafeURLError("only http/https URLs are allowed")
    if not parts.hostname:
        raise UnsafeURLError("URL has no host")
    if parts.port not in (None, 80, 443):
        raise UnsafeURLError("non-standard ports are not allowed")
    if parts.username or parts.password:
        raise UnsafeURLError("credentials in URLs are not allowed")

    try:
        infos = await asyncio.get_running_loop().getaddrinfo(parts.hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise UnsafeURLError("host could not be resolved") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global or ip.is_multicast:
            raise UnsafeURLError("host resolves to a non-public address")


async def _safe_get(url: str, accept_html_only: bool) -> tuple[str, str] | None:
    """GET a URL with SSRF checks on every redirect hop. Returns (final_url, text)."""
    client = get_http_client()
    for _ in range(MAX_REDIRECTS + 1):
        await assert_public_url(url)
        async with client.stream("GET", url) as response:
            if response.is_redirect and "location" in response.headers:
                url = urljoin(url, response.headers["location"])
                continue
            if response.status_code != 200:
                return None
            content_type = response.headers.get("content-type", "")
            if accept_html_only and "html" not in content_type:
                return None
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > MAX_HTML_BYTES:
                    break
            return url, body.decode(response.encoding or "utf-8", errors="replace")
    return None


_robots_cache: dict[str, RobotFileParser | None] = {}


async def _robots_allows(url: str) -> bool:
    parts = urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    if origin not in _robots_cache:
        parser: RobotFileParser | None = RobotFileParser()
        try:
            result = await _safe_get(origin + "/robots.txt", accept_html_only=False)
            parser.parse(result[1].splitlines() if result else [])  # no robots.txt -> allowed
        except (UnsafeURLError, httpx.HTTPError):
            parser = None  # could not check -> be conservative
        if len(_robots_cache) > 2000:
            _robots_cache.clear()
        _robots_cache[origin] = parser
    parser = _robots_cache[origin]
    return parser is not None and parser.can_fetch(USER_AGENT, url)


async def fetch_public_html(url: str) -> tuple[str, str] | None:
    """
    Safely download a public HTML page. Returns (final_url, html) or None when the
    page is unsafe, disallowed by robots.txt, not HTML, or unreachable.
    """
    try:
        if not await _robots_allows(url):
            logger.info("robots.txt disallows %s", urlsplit(url).netloc)
            return None
        return await _safe_get(url, accept_html_only=True)
    except UnsafeURLError as exc:
        logger.warning("Blocked unsafe URL %s: %s", urlsplit(url).netloc, exc)
    except httpx.HTTPError as exc:
        logger.info("Could not fetch %s: %s", urlsplit(url).netloc, type(exc).__name__)
    return None
