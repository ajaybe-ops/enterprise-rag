"""LLM access with token, latency and cost accounting.

Why this replaced the old `app/gateway/client.py`
-------------------------------------------------
That module returned `(content, cache_status)` and discarded the provider's
usage block, so cost and token observability were impossible — you cannot report
a number you never captured. It also gave the Portkey path two retries and the
degraded direct-OpenAI path none, which means the fallback was *less* reliable
than the primary. Both are fixed here:

* Token counts come from the provider's own `usage` object. When it is absent
  the count is estimated and `Usage.estimated` is set, so an estimate is never
  reported as a measurement.
* Retries are applied identically on both paths, and only for failures that can
  actually succeed on retry: connection errors, timeouts, 429 and 5xx. A 400 is
  a malformed request; retrying it just doubles the latency of a doomed call.
* Cost is an **estimate** from configurable list prices, labelled as such.

Prompt construction lives in `app/generation.py`, not here. This module does
transport, retries and accounting; it does not know what a RAG answer is.
"""

from __future__ import annotations

import json
import random
import time
from typing import Any, Sequence

import requests

from app.config import Settings
from app.types import Usage

__all__ = ["complete", "estimate_tokens", "LLMUnavailable", "approx_cost_usd"]

# Statuses worth retrying. 400/401/403/404 are not here on purpose.
_RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

# Very rough characters-per-token for English + code. Used ONLY when the
# provider reports no usage. Deliberately labelled an approximation in the
# dataclass so it is never mistaken for a provider count.
_CHARS_PER_TOKEN = 4.0


class LLMUnavailable(RuntimeError):
    """Raised when the LLM could not be reached or returned nothing usable."""


def estimate_tokens(text: str) -> int:
    """Approximate token count. Only used when the provider omits `usage`."""
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def approx_cost_usd(settings: Settings, usage: Usage) -> float:
    """Cost estimate in USD from configurable per-1M-token list prices.

    These are estimates against published list prices, not billing. Real spend
    depends on the negotiated rate, cached-token discounts and the fallback
    model's price. See docs/08_COST_MODEL.md for provenance and the drift caveat.
    """
    return (
        usage.prompt_tokens / 1_000_000 * settings.cost_per_mtok_input
        + usage.completion_tokens / 1_000_000 * settings.cost_per_mtok_output
    )


def _direct_openai(
    settings: Settings,
    messages: Sequence[dict],
    temperature: float,
    model: str,
    feature: str = "rag",
) -> dict:
    """OpenAI-compatible chat completion. Used directly or via Portkey.

    Both go through the same OpenAI-compatible HTTP surface: Portkey *is* an
    OpenAI-compatible proxy, so one code path with an optional `base_url` and
    extra headers. Two SDKs (one native, one LangChain) meant two retry
    behaviours, which is how the retry inconsistency happened.

    The URL is not cosmetic when Portkey is on. Posting Portkey headers to
    `api.openai.com` reaches OpenAI directly and the gateway silently does
    nothing: no virtual key, no `@slug/model` routing, no failover, no cache
    header. The call still returns 200, so the failure is invisible unless the
    base URL follows the gateway.
    """
    from app.gateway import gateway_headers, gateway_url

    base_url = gateway_url(settings) if settings.gateway_enabled else "https://api.openai.com/v1"
    headers = {"Content-Type": "application/json", **gateway_headers(settings, feature=feature)}

    body: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
    }

    last_error: Exception | None = None
    for attempt in range(max(1, settings.llm_max_retries + 1)):
        started = time.perf_counter()
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                data=json.dumps(body),
                timeout=settings.llm_timeout_s,
            )
            if response.status_code in _RETRY_STATUSES and attempt < settings.llm_max_retries:
                # Full jitter rather than fixed backoff: synchronised retries
                # against a struggling provider are how a brownout becomes an
                # outage.
                time.sleep(random.uniform(0, 2**attempt))
                continue
            response.raise_for_status()
            return {
                "json": response.json(),
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "cache_status": response.headers.get("x-portkey-cache-status", "").upper() or None,
            }
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= settings.llm_max_retries:
                break
            time.sleep(random.uniform(0, 2**attempt))

    raise LLMUnavailable(f"LLM call failed after {settings.llm_max_retries + 1} attempts: {last_error}")


def _extract_usage(payload: dict, model: str, latency_ms: float) -> tuple[str, Usage]:
    try:
        content = payload["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMUnavailable(f"malformed LLM response: {exc}") from exc

    raw_usage = payload.get("usage") or {}
    prompt_tokens = raw_usage.get("prompt_tokens")
    completion_tokens = raw_usage.get("completion_tokens")

    if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        usage = Usage(
            model=raw_usage.get("model") or model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            estimated=False,
        )
    else:
        # No usage block. Estimate from the request/response and mark it, so the
        # caller can exclude it from a cost total that claims to be measured.
        prompt_text = " ".join(str(m.get("content", "")) for m in payload.get("_request_messages", []))
        usage = Usage(
            model=model,
            prompt_tokens=estimate_tokens(prompt_text),
            completion_tokens=estimate_tokens(content),
            latency_ms=latency_ms,
            estimated=True,
        )

    return content, usage


def complete(
    settings: Settings,
    messages: Sequence[dict],
    *,
    feature: str = "rag",
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> tuple[str, Usage]:
    """Single chat completion. Returns `(content, usage)`.

    Raises `LLMUnavailable` if the model could not be reached. Callers decide
    whether that is fatal — the answer path turns it into an explicit "not
    configured" status rather than a 500.
    """
    if not settings.llm_enabled:
        raise LLMUnavailable(
            "no LLM configured: set PORTKEY_API_KEY or OPENAI_API_KEY"
        )

    if settings.gateway_enabled:
        target = f"@{settings.portkey_primary_slug}/{model or settings.openai_model}"
    else:
        target = model or settings.openai_model

    result = _direct_openai(settings, messages, temperature, target, feature=feature)
    payload = result["json"]
    # Preserve the request for the estimation path, then drop it: the returned
    # payload is logged at debug level, and the request may contain user data.
    payload["_request_messages"] = list(messages)
    try:
        content, usage = _extract_usage(payload, target, result["latency_ms"])
    finally:
        payload.pop("_request_messages", None)

    usage = Usage(
        model=usage.model,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        latency_ms=usage.latency_ms,
        cost_usd=approx_cost_usd(settings, usage),
        estimated=usage.estimated,
        cached=result["cache_status"] == "HIT",
    )
    return content, usage
