"""
Validators for contact information.

Rule: we only *extract* contact details that are already published in public
text. Every value is validated; anything doubtful is dropped. Nothing is ever
generated or guessed.
"""

import re

import phonenumbers

_EMAIL_RE = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,24}\b")

# Strings that look like emails but are not real contact addresses.
_FILE_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "svg", "webp", "css", "js", "ico", "bmp", "pdf"}
_PLACEHOLDER_DOMAINS = {
    "example.com", "example.org", "example.net", "domain.com", "yourdomain.com",
    "email.com", "mail.com", "test.com", "sentry.io", "wixpress.com", "sentry-next.wixpress.com",
}
_PLACEHOLDER_LOCAL_PARTS = {"name", "your", "yourname", "user", "username", "email", "someone", "john.doe", "test"}

# "WhatsApp: +971 50 123 4567" / "WhatsApp +1 (202) 555-0123"
_WHATSAPP_TEXT_RE = re.compile(r"whats\s?app\s*(?:us|me|no\.?|number|#)?\s*[:\-–]?\s*(\+?[\d][\d\s().-]{6,18}\d)", re.I)


def validate_email(raw: str) -> str | None:
    """Return a normalized email address, or None if it is invalid or a placeholder."""
    email = raw.strip().strip(".").lower()
    if email.startswith("mailto:"):
        email = email[7:].split("?")[0]
    if len(email) > 254 or not _EMAIL_RE.fullmatch(email):
        return None
    local, _, domain = email.rpartition("@")
    if ".." in email or local.startswith(".") or local.endswith("."):
        return None
    if domain.rsplit(".", 1)[-1] in _FILE_EXTENSIONS:  # e.g. logo@2x.png
        return None
    if domain in _PLACEHOLDER_DOMAINS or local in _PLACEHOLDER_LOCAL_PARTS:
        return None
    return email


def extract_emails(text: str) -> set[str]:
    found = set()
    for match in _EMAIL_RE.findall(text or ""):
        email = validate_email(match)
        if email:
            found.add(email)
    return found


def validate_phone(raw: str, region: str | None = None) -> str | None:
    """
    Validate a phone number with Google's libphonenumber rules and return it in
    E.164 format (+<country><number>). `region` (e.g. "US") is used when the
    number has no +country prefix.
    """
    try:
        number = phonenumbers.parse(raw, region.upper() if region else None)
    except phonenumbers.NumberParseException:
        return None
    if not phonenumbers.is_valid_number(number):
        return None
    return phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.E164)


def extract_phones(text: str, region: str | None = None, limit: int = 3) -> set[str]:
    """Find valid phone numbers written in free text (e.g. a bio or web page)."""
    found: set[str] = set()
    if not text:
        return found
    matcher = phonenumbers.PhoneNumberMatcher(
        text, region.upper() if region else None, leniency=phonenumbers.Leniency.VALID
    )
    for match in matcher:
        found.add(phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.E164))
        if len(found) >= limit:
            break
    return found


def extract_whatsapp_numbers(text: str, region: str | None = None) -> set[str]:
    """
    Numbers that the text *explicitly labels* as WhatsApp, returned as E.164.
    A plain phone number is NOT assumed to be on WhatsApp.
    """
    found = set()
    for raw in _WHATSAPP_TEXT_RE.findall(text or ""):
        phone = validate_phone(raw, region)
        if phone:
            found.add(phone)
    return found
