"""WhatsApp - publicly published WhatsApp contact links.

WhatsApp has no public directory or search API (by design). We find pages
where prospects publish a wa.me / api.whatsapp.com link, or a number explicitly
labelled "WhatsApp". A plain phone number is never assumed to be on WhatsApp.
"""

from app.models import RawProfile
from app.platforms.base import ContactSearchPlatform


class WhatsAppPlatform(ContactSearchPlatform):
    name = "WhatsApp"
    extra_terms = '("whatsapp" OR "wa.me")'

    def has_contact(self, profile: RawProfile) -> bool:
        return bool(profile.whatsapp)
