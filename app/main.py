"""
FastAPI application - entry point.

Run locally:   uvicorn app.main:app --port 4000
Docs:          http://localhost:4000/docs
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import get_settings
from app.schemas import ErrorResponse, ProspectRequest, ProspectResponse
from app.services.prospect_service import DiscoveryError, discover_prospects
from app.utils.http_client import close_http_client
from app.utils.rate_limiter import SlidingWindowRateLimiter

settings = get_settings()

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# HTTP libraries log full request URLs at INFO level; keep them quiet so
# nothing sensitive (e.g. query-string parameters) ends up in logs.
for noisy in ("httpx", "httpx2", "httpcore", "anthropic"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("app")

MAX_BODY_BYTES = 10_000


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting | LLM: %s | search: %s | YouTube API: %s",
                settings.llm_model if settings.llm_enabled else "disabled (rule-based fallback)",
                "enabled" if settings.search_enabled else "disabled",
                "enabled" if settings.youtube_api_enabled else "disabled")
    yield
    await close_http_client()


app = FastAPI(
    title="AI-Powered Multi-Platform Prospect Discovery API",
    version="1.0.0",
    description="Discovers public prospect profiles across Google, LinkedIn, X, YouTube, "
                "Instagram, WhatsApp, Facebook, Email and Threads from a natural-language query.",
    lifespan=lifespan,
)

rate_limiter = SlidingWindowRateLimiter(max_requests=settings.rate_limit_per_minute)


def error_response(status_code: int, code: str, message: str, query: str | None = None) -> JSONResponse:
    body = ErrorResponse(query=query, error={"code": code, "message": message})
    return JSONResponse(status_code=status_code, content=body.model_dump())


# --------------------------------------------------------------------------- #
# Middleware & error handlers - every error is returned as structured JSON
# --------------------------------------------------------------------------- #
@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    length = request.headers.get("content-length")
    if length and (not length.isdigit() or int(length) > MAX_BODY_BYTES):
        return error_response(413, "payload_too_large", "Request body is too large.")
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    if first.get("type") == "json_invalid":
        return error_response(422, "invalid_request", "Request body must be valid JSON.")
    field = ".".join(str(part) for part in first.get("loc", []) if part != "body") or "body"
    message = str(first.get("msg", "Invalid request")).removeprefix("Value error, ")
    return error_response(422, "invalid_request", f"{field}: {message}")


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(_: Request, exc: StarletteHTTPException):
    return error_response(exc.status_code, "http_error", str(exc.detail))


@app.exception_handler(Exception)
async def unexpected_error_handler(_: Request, exc: Exception):
    logger.exception("Unhandled error")  # full details go to the server log only
    return error_response(500, "internal_error", "An unexpected error occurred.")


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
async def root():
    return {"service": app.title, "docs": "/docs", "endpoint": "POST /prospect-discovery"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "llm_configured": settings.llm_enabled,
        "search_configured": settings.search_enabled,
        "youtube_api_configured": settings.youtube_api_enabled,
    }


@app.post(
    "/prospect-discovery",
    response_model=ProspectResponse,
    responses={
        413: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 429: {"model": ErrorResponse},
        500: {"model": ErrorResponse}, 502: {"model": ErrorResponse}, 503: {"model": ErrorResponse},
    },
)
async def prospect_discovery(payload: ProspectRequest, request: Request):
    """Discover public prospects matching a natural-language query."""
    client_id = request.client.host if request.client else "unknown"
    if not rate_limiter.allow(client_id):
        return error_response(429, "rate_limited",
                              f"Too many requests. Limit: {settings.rate_limit_per_minute} per minute.",
                              payload.query)
    try:
        return await discover_prospects(payload.query)
    except DiscoveryError as exc:
        return error_response(exc.status_code, exc.code, exc.message, payload.query)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "4000")))
