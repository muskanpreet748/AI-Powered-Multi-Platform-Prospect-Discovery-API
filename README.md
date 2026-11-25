# AI-Powered Multi-Platform Prospect Discovery API

A FastAPI service that takes a natural-language query such as
*"Football lovers in Washington DC in USA"*, understands the audience and location with an LLM,
discovers matching **public** profiles across 9 platforms, merges profiles that belong to the same
person, filters them for relevance, and returns structured JSON.

| | Link |
|---|---|
| **Live demo** | https://prospect-discovery-api.onrender.com |
| **API endpoint** | `POST https://prospect-discovery-api.onrender.com/prospect-discovery` |
| **API documentation** | https://prospect-discovery-api.onrender.com/docs |
| **Health check** | https://prospect-discovery-api.onrender.com/health |

> Hosted on Render's free tier: after 15 minutes without traffic the service sleeps, and the first
> request takes about a minute to wake it. Later requests respond normally (typically 20–40 s for a
> discovery, which searches 9 platforms).

```bash
curl -X POST https://prospect-discovery-api.onrender.com/prospect-discovery \
  -H "Content-Type: application/json" \
  -d '{"query": "Restaurant owners in Dubai UAE"}'
```

Locally:

```http
POST http://localhost:4000/prospect-discovery
{"query": "Football lovers in Washington DC in USA"}
```

**Core principle: nothing is invented.** Every name, URL, email and phone number in a response is
copied from a public search result, a public API, or the prospect's own public website, and is
then validated. The LLM only *reads and classifies* text. It never produces links or contact details.

---

## Contents
1. [Quick start](#1-quick-start)
2. [How it works](#2-how-it-works)
3. [Supported platforms and discovery sources](#3-supported-platforms-and-discovery-sources)
4. [API reference](#4-api-reference)
5. [Duplicate detection](#5-duplicate-detection-and-consolidation)
6. [Relevance evaluation](#6-relevance-evaluation)
7. [Contact validation](#7-public-contact-discovery-and-validation)
8. [Security](#8-security)
9. [LLM research and recommendation](#9-llm-platform--model-research)
10. [Testing](#10-testing)
11. [Deployment](#11-deployment)
12. [Limitations](#12-known-limitations)

---

## 1. Quick start

Requirements: Python 3.11+ (developed and tested on 3.12).

```bash
git clone https://github.com/muskanpreet748/AI-Powered-Multi-Platform-Prospect-Discovery-API.git
cd AI-Powered-Multi-Platform-Prospect-Discovery-API

python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then add your API keys to .env
uvicorn app.main:app --port 4000
```

Open **http://localhost:4000/docs** for interactive Swagger documentation.

| Variable | Required | Purpose |
|---|---|---|
| `SERPER_API_KEY` | **Yes** | Google search results via [serper.dev](https://serper.dev) (2,500 free queries) |
| `LLM_PROVIDER` | No | `anthropic` (default, recommended for production) or `gemini` (free tier) |
| `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` | Recommended | Key for the selected provider. Without one, a rule-based fallback is used. |
| `YOUTUBE_API_KEY` | Optional | Official YouTube Data API v3. Without it, YouTube is discovered via Google's index. |
| `LLM_MODEL` | No | Overrides the provider default (`claude-haiku-5-5` / `gemini-3.5-flash-lite`, see [section 9](#9-llm-platform--model-research)) |

Other tuning variables (timeouts, limits, cache TTL, rate limit) are documented in [.env.example](.env.example).

Run the five practical test queries against the running server and save the responses to `samples/`:

```bash
python scripts/run_test_queries.py
```

---

## 2. How it works

```text
POST /prospect-discovery {"query": "..."}
   │
   ├─ 1. Validate input (Pydantic: 3–300 chars, must contain words, no unknown fields)
   ├─ 2. Query understanding (LLM: Claude or Gemini, structured JSON output; cached)
   │      → interest, audience, city/state/country, target type, keywords, search terms
   │      (fallback: rule-based parser if the LLM is unavailable)
   ├─ 3. Search-query generation per platform (e.g. site:instagram.com football fan "Washington DC")
   ├─ 4. Parallel discovery on 9 independent platform modules
   │      (each with its own timeout; a failing platform is reported, not fatal)
   ├─ 5. Profile extraction: canonical profile URLs, names, public contacts, cross-links
   ├─ 6. Website enrichment: visit prospect websites (SSRF-safe, robots.txt-aware)
   │      to collect owner-published emails, phones, WhatsApp and social links
   ├─ 7. Duplicate detection (union-find on identity signals)
   ├─ 8. Relevance: deterministic rules + batched LLM review (40 candidates per call, in parallel)
   └─ 9. Pydantic-validated JSON response
```

### Project structure

```text
app/
├── main.py                    FastAPI app, endpoint, error handlers, body-size + rate limits
├── config.py                  Settings from environment variables (keys stored as SecretStr)
├── schemas.py                 Request/response models (public API contract)
├── models.py                  Internal dataclasses (RawProfile, ProspectGroup, SearchResult)
├── services/
│   ├── llm_service.py         LLM providers (Claude via Anthropic SDK, Gemini via REST): structured outputs, errors, usage logging
│   ├── query_service.py       Query understanding, rule-based fallback, search-query builder
│   ├── search_service.py      Serper (Google) client: caching, concurrency limit, retries
│   ├── profile_extractor.py   Search result / web page → profile + contacts
│   ├── deduplication_service.py  Entity resolution (union-find)
│   ├── relevance_service.py   Rule scoring + batched LLM review
│   └── prospect_service.py    Orchestrator
├── platforms/                 One module per platform, common interface (base.py)
│   ├── google.py  linkedin.py  x_platform.py  youtube.py  instagram.py
│   └── whatsapp.py  facebook.py  email_search.py  threads.py
└── utils/
    ├── validators.py          Email / phone / WhatsApp validation (libphonenumber)
    ├── normalizers.py         Profile-URL detection and canonicalisation, name cleaning
    ├── http_client.py         Shared client, retries with backoff, SSRF-safe page fetching
    ├── cache.py               In-memory TTL cache
    └── rate_limiter.py        Per-client sliding-window limiter
tests/test_prospect_discovery.py   57 offline tests (no network, no keys)
scripts/run_test_queries.py        Runs the 5 practical test queries
```

**Adding a platform** means writing one class in `app/platforms/` that implements
`availability()` and `discover()`, then adding it to the list in `app/platforms/__init__.py`.

---

## 3. Supported platforms and discovery sources

Only official APIs or an authorized search API are used. The service never logs in to a platform,
never scrapes a platform's own pages, and never accesses private profiles.

| Platform | Discovery method | What is returned | Why this method |
|---|---|---|---|
| **Google** | Serper.dev (Google organic results), excluding social networks and directories | Prospects' **own websites**, enriched with the contacts and social links published on them | Google is used as a discovery source. Listicles, directories, news and search URLs are filtered out. |
| **LinkedIn** | Google index, `site:linkedin.com/in` (plus `/company` for business queries) | Public profile / company page URLs | LinkedIn APIs do not allow member search, and scraping is prohibited |
| **X** | Google index, `site:x.com OR site:twitter.com` | Profile URLs (post URLs are mapped to their author) | User search requires a paid X API tier |
| **YouTube** | **YouTube Data API v3** (`search.list` type=channel + `channels.list`). Falls back to Google index without a key. | Channel URL, title, description, country | Official API available |
| **Instagram** | Google index, `site:instagram.com` | Public profile URLs | Graph API cannot search arbitrary users |
| **Facebook** | Google index, `site:facebook.com` | Public profiles and pages (groups, posts and events rejected) | Graph API does not search people |
| **Threads** | Google index, `site:threads.net OR site:threads.com` | Public profile URLs | Threads API is limited to the authenticated user |
| **WhatsApp** | Pages that publish `wa.me` / `api.whatsapp.com` links, or numbers explicitly labelled "WhatsApp" | `https://wa.me/<number>` links | WhatsApp has no public directory. A plain phone number is never assumed to be on WhatsApp. |
| **Email** | Search results and websites that publish an email address | `mailto:` links | Only published addresses. Nothing is guessed (no `firstname@company` patterns). |

Phone numbers are returned with `"platform": "Phone"` and a `tel:+E164` link, inside `social_link`
as the specification requires. A prospect's own website is returned with `"platform": "Google"`,
the platform through which it was discovered.

Each response includes a `platforms` array showing which sources ran (`ok`), failed (`failed`, with
the reason), or were not configured (`skipped`).

---

## 4. API reference

### `POST /prospect-discovery`

Request:

```json
{ "query": "Football lovers in Washington DC in USA" }
```

Response (200). The structure is shown below with an illustrative prospect. Real outputs for the five test
queries are saved in [`samples/`](samples/) by `scripts/run_test_queries.py`.

```json
{
  "success": true,
  "query": "Football lovers in Washington DC in USA",
  "total_prospects": 1,
  "prospects": [
    {
      "name": "<name from public profile>",
      "location": "Washington DC, USA",
      "social_link": [
        { "platform": "Instagram", "link": "https://www.instagram.com/<handle>/" },
        { "platform": "X", "link": "https://x.com/<handle>" },
        { "platform": "Email", "link": "mailto:<published address>" },
        { "platform": "WhatsApp", "link": "https://wa.me/<number>" }
      ],
      "relevance": {
        "score": 0.95,
        "status": "confirmed",
        "interest_match": true,
        "location_match": true,
        "evidence": ["profile text mentions 'football'", "profile text mentions location 'Washington DC'",
                     "AI review: bio says lifelong Commanders fan in DC"],
        "evaluated_by": "rules+llm"
      },
      "sources": ["https://www.instagram.com/<handle>/", "https://x.com/<handle>"]
    }
  ],
  "understanding": {
    "is_valid_query": true,
    "interest": "Football",
    "audience": "Football lovers",
    "location": { "city": "Washington DC", "state": "District of Columbia", "country": "USA", "country_code": "US" },
    "target_type": "individual",
    "keywords": ["football", "nfl", "soccer"],
    "search_terms": ["football fan", "nfl fan"],
    "location_aliases": ["DC", "D.C.", "Washington, D.C."]
  },
  "understanding_source": "llm",
  "platforms": [
    { "platform": "Google", "status": "ok", "profiles_found": 6, "message": null },
    { "platform": "Facebook", "status": "failed", "profiles_found": 0, "message": "timed out after 25s" }
  ]
}
```

The fields required by the specification (`success`, `query`, `total_prospects`, `prospects[].name`,
`location`, `social_link[]`) are unchanged. The extra fields exist so that verified and
inferred information can be told apart:

- `relevance.evidence` lists the text that justified the match.
- `location` is set **only** when public text supports it. Otherwise it is `null`.
- `relevance.status` is `confirmed` when the evidence covers both interest and location, and `likely` when it covers the interest only.
- `sources` lists the public pages each prospect was found on, so every result can be checked.

### Errors

All errors use one JSON shape: `{"success": false, "query": ..., "error": {"code", "message"}}`.

| Status | Code | When |
|---|---|---|
| 413 | `payload_too_large` | Body over 10 KB |
| 422 | `invalid_request` | Missing/empty/too long query, no words, unknown fields, invalid JSON |
| 422 | `invalid_query` | The LLM judged that the text does not describe people or businesses to find |
| 429 | `rate_limited` | More than `RATE_LIMIT_PER_MINUTE` requests per minute from one client |
| 502 | `all_sources_failed` | Every discovery platform failed |
| 503 | `no_sources_configured` | No search API key configured |
| 500 | `internal_error` | Unexpected error (details are logged server-side and never returned) |

Other endpoints: `GET /health` (configuration status, no secrets) and `GET /docs` (Swagger UI).

---

## 5. Duplicate detection and consolidation

`deduplication_service.py` links two profiles only on a **strong identity signal**:

1. **Same canonical URL.** The same profile was found by different queries. URLs are canonicalised first (`m.facebook.com/x/` and `www.facebook.com/x` are the same).
2. **Same email, phone or WhatsApp number.**
3. **Explicit cross-link.** A website or bio links to that profile, for example a restaurant's site linking its Instagram.
4. **Same username on two platforms AND a similar display name** (SequenceMatcher ratio ≥ 0.85, or every word of the shorter name appears in the longer one).

**A similar name alone never merges.** Two "John Smith" profiles stay separate, and the same
handle with a different name also stays separate. This is covered by unit tests.

Linked profiles are grouped with a **union-find** structure, so if A~B and B~C then all three
become one prospect. The display name is taken from the most authoritative profile in the group
(LinkedIn → Facebook → Instagram → …).

---

## 6. Relevance evaluation

Relevance is decided in two layers (`relevance_service.py`):

1. **Rules (free, deterministic).** Whole-word matches of the LLM-generated keywords and of the city, state, aliases and country in the profile's public text. YouTube's channel country counts as country-level evidence.
2. **LLM review, in batches of 40 candidates (up to 120) that run in parallel.** The LLM receives only the collected public text and returns, per candidate ID, `audience_match`, `location_match` and `genuine_prospect` (false for news, listicles, directories, job ads and events), plus a short reason. Candidate text is placed in delimited tags and treated as untrusted data. Because the LLM reads context, it can **overrule a plain-text location match**: a founder whose surname is "London", or someone who only visited Dubai, is not marked as located there.

A prospect is **kept only if the evidence supports the interest or audience**. A match on name or
location alone is never enough. Score: 0.6 for interest, +0.3 for local location (or +0.15 for
country), and up to +0.1 when several profiles or contacts corroborate each other.

If the LLM is unavailable, the rule layer alone decides, and every prospect is marked `"evaluated_by": "rules"`.

---

## 7. Public contact discovery and validation

| Type | Extraction | Validation |
|---|---|---|
| Email | `mailto:` links and text on public pages and snippets | Regex + rejection of placeholders (`example.com`), image names (`logo@2x.png`) and tracking domains |
| Phone | `tel:` links and phone numbers in public text | Google **libphonenumber**: must be a valid number for the target country, returned as E.164 |
| WhatsApp | `wa.me` / `api.whatsapp.com/send` links, or numbers explicitly labelled "WhatsApp" | Same phone validation; returned as `https://wa.me/<digits>` |

Association rules:
- If a page lists more than 3 different emails or phones, it is treated as a directory, and none of its contacts are attached.
- A social link on a website is used only when the site links to exactly **one** account on that platform. Several links (partners, the web designer) are ambiguous and skipped.

---

## 8. Security

| Risk | Mitigation |
|---|---|
| **SSRF** (URLs from search results are untrusted) | Only `http`/`https` on ports 80/443; no credentials in URLs; the hostname is resolved and **private, loopback, link-local and metadata IPs are rejected**; redirects are followed manually and every hop is re-checked; 1.5 MB download cap; HTML only (`utils/http_client.py`) |
| **Prompt injection** | User queries and scraped text sit inside delimited tags, with angle brackets escaped. System prompts tell the model to treat that content as data. Output is constrained to a JSON schema and validated, and the LLM can only return booleans or labels for IDs it was given. |
| **Search-query injection** | LLM- and user-derived terms are sanitised (quotes and operators such as `site:` removed) before being placed into search queries |
| **Secret leakage** | Keys come only from environment variables, are stored as `SecretStr`, and `.env` is git-ignored. The YouTube key is sent in a header, not the URL. HTTP library logging is raised to WARNING so request URLs are not logged. Error responses never include stack traces. |
| **Abuse / cost attacks** | Per-client rate limit (default 10 per minute), 10 KB body limit, 300-character query limit, `extra="forbid"` on the request model, caps on queries per platform, results per query, enriched websites, LLM candidates and returned prospects |
| **Responsible data use** | Public data only; robots.txt is honoured; no login or authentication bypass; nothing is stored beyond a short in-memory cache; every prospect lists its `sources` |
| **Container** | Runs as a non-root user; `.env` excluded via `.dockerignore` |

---

## 9. LLM platform & model research

The LLM does two small jobs per request:
1. **Query understanding.** Extract structured parameters from one sentence (one call).
2. **Relevance classification.** Judge short profile texts, 40 per call (1–3 parallel calls).

Both are short, schema-bound extraction and classification tasks. They need reliable JSON, low
latency and low cost far more than deep reasoning, which points to a provider's small, fast tier.

The service supports two providers behind one interface (`LLM_PROVIDER`): **Anthropic Claude**
(official SDK) and **Google Gemini** (REST). Both use JSON-schema structured outputs validated into
the same Pydantic models.

### A. Comparison

Prices are standard-tier USD per 1M tokens, read from the official pricing pages on **25 Nov 2026**
([Anthropic](https://platform.claude.com/docs/en/about-claude/pricing),
[OpenAI](https://developers.openai.com/api/docs/pricing),
[Google](https://aistudio.google.com/docs/pricing)).
Cost per request uses the **measured** average token usage from section D (≈5,060 input / ≈3,350 output
tokens). Token counts were measured on Gemini; other tokenizers differ somewhat, so the costs for
other providers are estimates.

| Provider | Model | Input / Output price | Cost per request | Context window | Structured JSON | Speed | Suitability |
|---|---|---|---|---|---|---|---|
| Anthropic | **Claude Haiku 5.5** (`claude-haiku-5-5`) | $0.10 / $0.50 (prompts ≤100K) | **≈ $0.0022** | 1M | JSON-schema structured outputs; SDK validates into Pydantic | Fast tier (estimated) | **Recommended for production** |
| Anthropic | Claude Sonnet 5.5 (`claude-sonnet-5-5`) | $2.00 / $10.00 | ≈ $0.044 | 1M | Same | Slower than Haiku (estimated) | Overkill for extraction (≈20× the cost) |
| OpenAI | GPT-6 Luna (`gpt-6-luna`) | $0.10 / $0.50 | ≈ $0.0022 | Not stated on pricing page | JSON-schema structured outputs | Fast tier (estimated) | Strong alternative |
| OpenAI | GPT-5.4 mini (`gpt-5.4-mini`) | $0.75 / $4.50 | ≈ $0.019 | Not stated on pricing page | JSON-schema structured outputs | Fast tier (estimated) | ~9× the cost, no benefit here |
| Google | **Gemini 3.5 Flash-Lite** (`gemini-3.5-flash-lite`) | $0.30 / $2.50; **free tier** | ≈ $0.0099 (free tier: $0) | Not stated on pricing page | `responseJsonSchema` | **Measured:** 1.2–2.3 s for a short extraction | **Implemented**; used for the free demo |
| Google | Gemini 3.1 Flash-Lite (`gemini-3.1-flash-lite`) | $0.25 / $1.50; free tier | ≈ $0.0063 | Not stated on pricing page | `responseJsonSchema` | **Measured:** 3.9–75 s on the free tier (very variable) | Too slow on the free tier |
| Google | Gemini 2.5 Flash-Lite | $0.10 / $0.40 | — | — | — | — | **Not available to new users** (API returns 404) |

**Measured vs estimated.** Only the Gemini models could be measured, because they were the
providers with keys available for this submission. The Gemini figures come from direct API calls and
from the server log of the five test queries ([`samples/server_log_test_queries.txt`](samples/server_log_test_queries.txt)).
The Anthropic and OpenAI speed entries are the providers' tier positioning, not measurements. The
service logs every LLM call's real latency and token usage
(`LLM RelevanceBatch | gemini-3.5-flash-lite | 6402 ms | input=2688 output=1688 tokens`), so the same
numbers can be collected for any provider by switching `LLM_PROVIDER`.

**Observed quality** (Gemini 3.5 Flash-Lite, five test queries): every query was understood correctly,
for example "Washington DC" → District of Columbia / `US` with aliases "DC" and "D.C.". The relevance
review removed news pages, listicles and unrelated accounts that the keyword rules alone had kept, and
it corrected misleading location matches. 3.1 Flash-Lite returned a weaker parse of the same query
(city "Washington", state "DC").

### B. Best model selection

| Need | Why Claude Haiku 5.5 fits |
|---|---|
| Query understanding | Short extraction with geography completion. Well within a small model's ability; verified on the free-tier equivalent (Gemini 3.5 Flash-Lite). |
| Search-query generation | Synonyms and keywords ("restaurant" → "restaurateur", "hospitality"). Templates combine them deterministically, so the LLM never writes raw search syntax. |
| Relevance classification | Boolean judgements from short texts, batched 40 per call |
| Structured JSON | Native JSON-schema structured outputs. The Anthropic SDK's `messages.parse()` returns a validated Pydantic object, so there is no fragile string parsing. |
| Speed | Small, fast tier with low effort configured |
| Cost | Lowest price tier among current-generation models: ≈4.5× cheaper than Gemini 3.5 Flash-Lite at paid rates, and tied with GPT-6 Luna |
| Reliability | During testing, the Gemini **free tier** returned HTTP 503 "overloaded" on 8 of 14 calls (all recovered through retries). A production service needs a paid tier anyway, and on paid tiers Haiku 5.5 is the cheapest. |

### C. Cost optimization (implemented)

| Technique | Where |
|---|---|
| **Batched relevance calls**: 40 candidates per call instead of one call per profile (≈40× fewer calls), run in parallel | `relevance_service.evaluate` |
| **Deterministic work stays deterministic**: URL detection, contact extraction, phone and email validation and deduplication use regex, libphonenumber and union-find, not the LLM | `utils/`, `deduplication_service` |
| **Rule pre-ranking and a cap**: at most 120 candidates reach the LLM, best rule score first | `MAX_LLM_RELEVANCE_CANDIDATES` |
| **Compact prompts**: candidate text truncated to 500 characters; short system prompts | `relevance_service`, `query_service` |
| **Caching**: query understanding and search results cached for 1 hour, so repeated queries cost nothing. Rule-based fallback answers are *not* cached, so the LLM is retried after an outage. | `utils/cache.py`, `query_service` |
| **Low effort setting** on Claude for these simple tasks (fewer reasoning tokens) | `llm_service.py` |
| **Small model** for the whole pipeline | `LLM_MODEL` |
| **Graceful degradation**: retries with backoff on overloads, then the rule-based path instead of failing the request | `http_client`, `query_service`, `relevance_service` |

Not applied, and why: *prompt caching* (the system prompts are shorter than the minimum cacheable
prefix) and the *Batch API* (50% cheaper, but asynchronous, which doesn't fit a synchronous endpoint).

**Next saving identified from the measurements:** the per-candidate `reason` text accounts for most
of the output tokens, and output tokens cost 5× more than input. Shortening it, or returning it only
for kept prospects, would cut LLM cost further.

**Note:** at these prices the **search API, not the LLM, is the dominant cost**. About 16 Serper
searches per request at ≈$0.001 each is ≈$0.016, against ≈$0.002 for Claude Haiku 5.5. That is why
search results are cached and queries per platform are capped (`MAX_QUERIES_PER_PLATFORM`). YouTube
Data API usage is about 201 quota units per request (10,000 free per day).

### D. Final recommendation

| Item | Value |
|---|---|
| **Recommended provider** | Anthropic |
| **Recommended model** | Claude Haiku 5.5 (`claude-haiku-5-5`), `LLM_PROVIDER=anthropic` |
| **Input tokens / request (measured)** | 2,306–8,333, average ≈ 5,060 (understanding ≈ 215; each relevance batch ≈ 650–4,000) |
| **Output tokens / request (measured)** | 1,877–5,653, average ≈ 3,350 (understanding ≈ 180; each relevance batch ≈ 290–2,500) |
| **LLM cost / request** | 5,060 × $0.10/1M + 3,350 × $0.50/1M ≈ **$0.0022** (≈ $2.20 per 1,000 requests) |
| **Response time (measured on Gemini 3.5 Flash-Lite)** | Query understanding: median 2.2 s (1.5–14.6 s including overload retries). Relevance batch: median 6.4 s (1.8–20.3 s). Whole request: 21–41 s, mostly spent searching 9 platforms. Haiku 5.5 was not measured. |
| **Why** | Cheapest current-generation tier, 1M context, schema-enforced JSON with Pydantic validation in the official SDK, and enough capability for extraction and classification |
| **Fallback model** | Gemini 3.5 Flash-Lite, already implemented: set `LLM_PROVIDER=gemini`. It also powers the free demo deployment. GPT-6 Luna is an equally priced alternative vendor. If the active LLM fails, the built-in rule-based parser and scorer keep the API working. |
| **Balancing approach** | Use the small model only where language understanding adds value (intent, context-aware relevance). Do everything else deterministically. Batch, cache and cap. Move to a larger model only if a labelled evaluation shows the small one misclassifies. |

---

## 10. Testing

```bash
pytest -q
```

57 offline tests (no network, no API keys) cover:
- email, phone and WhatsApp validation, including dropping numbers from other countries
- profile-URL detection and rejection of posts, groups, search pages and invalid numbers
- names: post text is never used as a name; "Contact …" prefixes are removed
- rule-based query understanding, LLM-failure fallback, and not caching fallback answers
- both LLM providers' error handling (blocked, cut off, invalid JSON, bad key, rate limit)
- search-query injection
- deduplication: same name not merged; merging by shared contact, cross-link, and handle plus name
- relevance: the LLM rejecting non-genuine profiles and overruling misleading location matches, and batched review of all candidates
- the SSRF guard
- input validation
- an end-to-end API run with fake search results: consolidation, website enrichment, listicle filtering
- isolation of a single platform failure, and the all-sources-failed path

### Live results (25 Nov 2026, Gemini 3.5 Flash-Lite, all platforms enabled)

Produced by `python scripts/run_test_queries.py`. Full JSON responses are in [`samples/`](samples/)
and the server log is in [`samples/server_log_test_queries.txt`](samples/server_log_test_queries.txt).

| Query | Raw profiles found | After merging duplicates | Relevant prospects returned | Server time |
|---|---:|---:|---:|---:|
| Football lovers in Washington DC in USA | 80 | 69 | 22 | 19.9 s |
| Digital marketing agency owners in New York USA | 107 | 93 | 50 (cap: `MAX_PROSPECTS`) | 40.8 s |
| Startup founders in London UK | 37 | 37 | 10 | 28.2 s |
| Cricket lovers in Ahmedabad India | 33 | 31 | 7 | 21.1 s |
| Restaurant owners in Dubai UAE | 49 | 48 | 18 | 30.0 s |

The gap between "after merging" and "relevant" is the relevance filter removing profiles that the
public text does not support (news pages, unrelated accounts, wrong audience).

Every returned prospect in these runs was reviewed by the LLM (`"evaluated_by": "rules+llm"`). Results
change between runs because search indexes change.

Screenshots of the Swagger UI and responses are in [`screenshots/`](screenshots/).

---

## 11. Deployment

**Docker**

```bash
docker build -t prospect-discovery .
docker run -p 4000:4000 --env-file .env prospect-discovery
```

**Render.com (free tier)**: New → Blueprint → select this repository ([render.yaml](render.yaml)).
The blueprint sets `LLM_PROVIDER=gemini` for a free deployment. Enter `SERPER_API_KEY`,
`GEMINI_API_KEY` and `YOUTUBE_API_KEY` in the dashboard. For production, set `LLM_PROVIDER=anthropic`
and `ANTHROPIC_API_KEY` instead. The health check is `/health` and the docs are at
`https://<your-app>.onrender.com/docs`.

---

## 12. Known limitations

- **Coverage depends on Google's public index.** LinkedIn, Instagram, Facebook, X and Threads offer no permitted people-search API, so only profiles that are public and indexed can be found. Snippets can be slightly out of date.
- **Location is often absent** from public snippets. Such prospects are returned with `location: null` and `status: "likely"` rather than having a location guessed.
- **WhatsApp** contacts exist only where someone published a WhatsApp link or labelled number, so many results have none.
- **The Gemini free tier is often overloaded** (HTTP 503). Retries absorbed every overload in the test run, but when retries are exhausted the request falls back to rules for that step. A paid tier avoids this.
- **The cache and rate limiter are in-memory** (per process). A multi-instance deployment would move them to Redis.
- **DNS rebinding:** the SSRF guard validates the resolved IPs before each request. A hostile DNS server could still answer differently on the connection itself. Fully closing this needs IP pinning at the connection layer or an egress proxy.
- **No labelled accuracy benchmark.** Relevance quality is protected by evidence requirements and schema validation, but it has not been measured against a hand-labelled set.
