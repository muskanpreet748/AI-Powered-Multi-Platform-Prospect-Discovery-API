"""
Platform registry. To support a new platform, implement a `BasePlatform`
subclass in its own module and add an instance to the list below.
"""

from app.platforms.base import BasePlatform
from app.platforms.email_search import EmailPlatform
from app.platforms.facebook import FacebookPlatform
from app.platforms.google import GooglePlatform
from app.platforms.instagram import InstagramPlatform
from app.platforms.linkedin import LinkedInPlatform
from app.platforms.threads import ThreadsPlatform
from app.platforms.whatsapp import WhatsAppPlatform
from app.platforms.x_platform import XPlatform
from app.platforms.youtube import YouTubePlatform


def get_platforms() -> list[BasePlatform]:
    return [
        GooglePlatform(),
        LinkedInPlatform(),
        XPlatform(),
        YouTubePlatform(),
        InstagramPlatform(),
        WhatsAppPlatform(),
        FacebookPlatform(),
        EmailPlatform(),
        ThreadsPlatform(),
    ]
