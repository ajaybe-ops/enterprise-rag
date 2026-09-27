"""
API safety: bearer-key authentication and rate limiting.

Rate limiting is a fixed 60s window counted in Upstash Redis when configured, so
the budget is shared across every replica. Without Redis it falls back to a
process-local window — correct for a single worker, per-process for many.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, status

from app import observability as logfire  # vendor-free tracing shim

from app.config import get_settings

_attempts: dict[str, deque[float]] = defaultdict(deque)
_lock = threading.Lock()
WINDOW_SECONDS = 60


def _client_id(api_key: object, forwarded_for: object) -> str:
    # FastAPI replaces the Header defaults with real values; a direct Python
    # call (tests, scripts) leaves the Header object in place. Coerce both.
    key = api_key if isinstance(api_key, str) else None
    if key:
        return f"key:{key[-8:]}"
    forwarded = forwarded_for if isinstance(forwarded_for, str) else None
    return f"ip:{forwarded or 'unknown'}"


def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> str | None:
    """Validate the API key when RAG_API_KEY is configured.

    Accepts either `X-API-Key: <key>` or `Authorization: Bearer <key>`.
    When no key is configured the app is open (local development) and this is a
    no-op.
    """
    if not get_settings().auth_enabled:
        return None

    provided = x_api_key
    if not provided and authorization and authorization.lower().startswith("bearer "):
        provided = authorization.split(" ", 1)[1].strip()

    if not provided or provided != get_settings().api_key:
        logfire.warning("🔒 Rejected request with missing/invalid API key.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key.",
        )
    return provided


def enforce_rate_limit(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_forwarded_for: str | None = Header(default=None, alias="X-Forwarded-For"),
) -> None:
    """Sliding 60s window limit of RATE_LIMIT_PER_MINUTE requests.

    In-process only. The previous implementation consulted a shared Redis
    counter first and silently fell back to a local one, which meant a
    deployment's effective rate limit depended on whether an unrelated cache
    happened to be reachable. A rate limit whose enforcement changes when a
    cache is down is not a rate limit.

    The consequence is stated plainly: across N workers or replicas the effective
    limit is N * `rate_limit_per_minute`, because each keeps its own window. A
    shared counter is the correct fix for a multi-process deployment and should
    be added deliberately -- not as a cache that may or may not answer.
    """
    limit = get_settings().rate_limit_per_minute
    if limit <= 0:
        return

    client = _client_id(x_api_key, x_forwarded_for)
    now = time.time()

    with _lock:
        window = _attempts[client]
        while window and now - window[0] > WINDOW_SECONDS:
            window.popleft()
        if len(window) >= limit:
            logfire.warning("Rate limit exceeded", client=client, count=len(window))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {limit} requests/minute.",
                headers={"Retry-After": str(WINDOW_SECONDS)},
            )
        window.append(now)
