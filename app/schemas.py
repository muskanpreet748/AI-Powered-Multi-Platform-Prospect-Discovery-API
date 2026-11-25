"""
Pydantic schemas: the shape of data entering and leaving the API.

Pydantic validates every field automatically. If a request does not match
`ProspectRequest`, FastAPI rejects it before our code runs. The response
models guarantee that the JSON we return always has the documented structure.
"""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Platform names exactly as defined in the assessment ("Phone" is used for
# phone numbers, which the spec requires inside `social_link` as well).
PlatformName = Literal[
    "Google", "LinkedIn", "X", "YouTube", "Instagram",
    "WhatsApp", "Facebook", "Email", "Threads", "Phone",
]

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #
class ProspectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # unknown fields -> 422 error

    query: str = Field(
        ...,
        min_length=3,
        max_length=300,
        examples=["Football lovers in Washington DC in USA"],
    )

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        value = _CONTROL_CHARS.sub(" ", value)
        value = " ".join(value.split())  # collapse repeated whitespace
        if len(value) < 3:
            raise ValueError("query must contain at least 3 characters")
        if not re.search(r"[^\W\d_]{2,}", value):  # needs at least one real word
            raise ValueError("query must describe an audience in words")
        return value


# --------------------------------------------------------------------------- #
# Query understanding (also used as the LLM's structured-output schema)
# --------------------------------------------------------------------------- #
class Location(BaseModel):
    city: str | None = Field(None, description="City name, e.g. 'Washington DC'. null if not given.")
    state: str | None = Field(None, description="State/province/region, e.g. 'District of Columbia'. null if unknown.")
    country: str | None = Field(None, description="Country name as commonly written, e.g. 'USA'. null if not given.")
    country_code: str | None = Field(None, description="ISO 3166-1 alpha-2 code, e.g. 'US'. null if unknown.")


class QueryUnderstanding(BaseModel):
    is_valid_query: bool = Field(
        description="true only if the query describes a group of people or businesses to find."
    )
    interest: str = Field(description="Core interest, profession or industry, e.g. 'Football'.")
    audience: str = Field(description="Target audience as a short phrase, e.g. 'Football lovers'.")
    location: Location
    target_type: Literal["individual", "business", "any"] = Field(
        description="'individual' for people, 'business' for companies/owners' businesses, 'any' if unclear."
    )
    keywords: list[str] = Field(
        description="3-8 lowercase words/phrases whose presence in a profile shows relevance "
        "(e.g. 'football', 'nfl', 'soccer fan')."
    )
    search_terms: list[str] = Field(
        description="2-3 short web-search phrases (2-4 words each, no location, no operators) "
        "to find these prospects, e.g. 'football fan', 'nfl fan'."
    )
    location_aliases: list[str] = Field(
        description="Other common spellings of the city/region, e.g. 'DC', 'D.C.', 'Washington, D.C.'."
    )


# --------------------------------------------------------------------------- #
# Response
# --------------------------------------------------------------------------- #
class SocialLink(BaseModel):
    platform: PlatformName
    link: str


class Relevance(BaseModel):
    score: float = Field(ge=0, le=1)
    status: Literal["confirmed", "likely"] = Field(
        description="'confirmed' = public text supports both interest and location; "
        "'likely' = supports the interest/audience only."
    )
    interest_match: bool
    location_match: bool
    evidence: list[str] = Field(description="Short reasons taken from the public profile text.")
    evaluated_by: Literal["rules", "rules+llm"]


class Prospect(BaseModel):
    name: str
    location: str | None = None
    social_link: list[SocialLink]
    relevance: Relevance
    sources: list[str] = Field(
        default_factory=list,
        description="Public pages where this prospect's information was found (for verification).",
    )


class PlatformStatus(BaseModel):
    platform: str
    status: Literal["ok", "failed", "skipped"]
    profiles_found: int = 0
    message: str | None = None


class ProspectResponse(BaseModel):
    success: bool = True
    query: str
    total_prospects: int
    prospects: list[Prospect]
    understanding: QueryUnderstanding | None = None
    understanding_source: Literal["llm", "rule_based"] | None = None
    platforms: list[PlatformStatus] = Field(default_factory=list)


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    success: bool = False
    query: str | None = None
    error: ErrorDetail
