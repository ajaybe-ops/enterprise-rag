"""API-level controls: key authentication and rate limiting.

Scope note
----------
These are transport-level controls on the HTTP surface. They are *not* the
authorization model: tenant scoping is `security.access`, and it runs against
verified claims in a real deployment. A valid API key means "you may ask
questions", never "you may read this tenant's documents".
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, status

from enterprise_rag.config.settings import get_settings
from enterprise_rag.observability import tracing

__all__ = ["enforce_rate_limit", "rate_limit_state", "reset_rate_limits", "verify_api_key"]

_attempts: dict[str, deque[float]] = defaultdict(deque)
_lock = threading.Lock()
WINDOW_SECONDS = 60


def _client_id(api_key: object, forwarded_for: object) -> str:
    """Identify the caller for rate-limiting purposes.

    Prefers the API key, because a key identifies a principal while an IP
    identifies a NAT gateway that may be serving hundreds of legitimate users.
    Falls back to the forwarded address only when the operator has declared a
    trusted proxy in front -- see `_forwarded_for`.
    """
    key = api_key if isinstance(api_key, str) else None
    if key:
        # Only the tail is used, so the limiter cannot be used to probe keys, and
        # the raw key never lands in a log line or an error message.
        return f"key:{key[-8:]}"
    return f"ip:{_forwarded_for(forwarded_for)}"


def _forwarded_for(forwarded_for: object) -> str:
    """The client address, honouring `X-Forwarded-For` only when configured.

    The header is client-controlled: anyone can send
    `X-Forwarded-For: <fresh-ip>` on every request and get an unlimited budget
    unless something upstream overwrites it. Trusting it unconditionally is the
    single easiest way to ship a rate limit that does not limit anything, so it
    is opt-in and defaults to off.
    """
    settings = get_settings()
    value = forwarded_for if isinstance(forwarded_for, str) else None
    if settings.trust_forwarded_for and value:
        # Left-most entry is the original client; the rest are proxies.
        return value.split(",")[0].strip()
    return "unknown"


def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> str | None:
    """Validate the API key when `RAG_API_KEY` is configured.

    Accepts `X-API-Key: <key>` or `Authorization: Bearer <key>`. With no key
    configured the service is open, which is the documented local-development
    mode -- and `/health` reports `auth: disabled` so the state is never silent.
    """
    settings = get_settings()
    if not settings.auth_enabled:
        return None

    provided = x_api_key
    if not provided and authorization and authorization.lower().startswith("bearer "):
        provided = authorization.split(" ", 1)[1].strip()

    # Compared with `secrets.compare_digest` rather than `!=`: a short-circuit
    # string comparison returns as soon as it finds a difference, which leaks the
    # length of the matching prefix through response timing. The key is the only
    # thing standing between the service and an open corpus.
    if not provided or not secrets_compare(provided, settings.api_key or ""):
        tracing.warning("rejected request with missing or invalid API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key.",
        )
    return provided


def secrets_compare(a: str, b: str) -> bool:
    """Constant-time comparison, with a length guard that does not leak more."""
    import secrets

    return secrets.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def enforce_rate_limit(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_forwarded_for: str | None = Header(default=None, alias="X-Forwarded-For"),
) -> None:
    """Sliding 60s window of `RATE_LIMIT_PER_MINUTE` requests per client.

    In-process only, and that is a deliberate choice rather than a gap. The
    previous implementation consulted a shared Redis counter and silently fell
    back to a local one, so a deployment's effective limit depended on whether an
    unrelated cache happened to be reachable -- a rate limit whose enforcement
    changes when a cache is down is not a rate limit.

    The consequence is stated rather than hidden: across N workers or replicas
    the effective limit is N * `rate_limit_per_minute`, because each keeps its
    own window. A shared counter is the correct fix for a multi-process
    deployment and should be added deliberately.
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
            tracing.warning("rate limit exceeded", client=client, count=len(window))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {limit} requests/minute.",
                headers={"Retry-After": str(WINDOW_SECONDS)},
            )
        window.append(now)


def rate_limit_state() -> dict[str, int]:
    """Current window occupancy per client. Reported by ``/metrics``."""
    with _lock:
        return {client: len(window) for client, window in _attempts.items() if window}


def reset_rate_limits() -> None:
    """Clear all windows. Test and CLI support."""
    with _lock:
        _attempts.clear()
