"""Email - publicly published email addresses.

Searches for pages where matching prospects publish an email address and keeps
only results whose public text actually contains a valid address.
(Named `email_search.py` to avoid clashing with Python's built-in `email` module.)
"""

from app.models import RawProfile
from app.platforms.base import ContactSearchPlatform


class EmailPlatform(ContactSearchPlatform):
    name = "Email"
    extra_terms = '("email" OR "contact" OR "@gmail.com")'

    def has_contact(self, profile: RawProfile) -> bool:
        return bool(profile.emails)
