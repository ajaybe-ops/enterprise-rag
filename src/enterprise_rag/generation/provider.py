"""Generation provider: transport, retries, and token/cost accounting.

Why a single OpenAI-compatible endpoint
---------------------------------------
The previous version had two code paths -- a Portkey gateway and a direct
OpenAI call -- and they were not equivalent. The gateway path got two retries
and the direct path got none, so the *fallback* was less reliable than the
primary, which is exactly backwards. It also had a latent bug: Portkey is
itself OpenAI-compatible, but the code posted gateway headers to
`api.openai.com` unless a base URL was threaded through every call. That
reaches OpenAI directly, the gateway silently does nothing, and the call
still returns 200 -- an invisible failure.

Every provider worth using here (OpenAI, vLLM, Ollama, Together, LiteLLM, a
corporate gateway) speaks the same `POST /chat/completions` shape. One path
parameterised by `llm_base_url` removes the divergence instead of documenting
it, and pointing `LLM_BASE_URL` at a local server makes the whole system
runnable with no external account at all.

What this module does *not* do
-----------------------------
It does not build prompts and it does not know what a RAG answer is. Prompt
construction lives in `enterprise_rag.context`; verification of whatever comes
back lives in `enterprise_rag.citations`. Keeping transport separate from
policy is what makes the extractive fallback a one-line substitution rather
than a second implementation of the answer path.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Sequence
from typing import Any

import requests

from enterprise_rag.config.settings import Settings
from enterprise_rag.types import Usage

__all__ = [
    "LLMUnavailable",
    "approx_cost_usd",
    "complete",
    "estimate_tokens",
    "token_stats",
]

# Statuses worth retrying. 400/401/403/404 are excluded on purpose: they are
# either a malformed request or a credential problem, and retrying either just
# doubles the latency of a doomed call while looking like resilience.
_RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

# Characters per token for English prose and code. Used ONLY when the provider
# omits its `usage` block, and every such count is flagged `estimated=True` so
# a cost total can exclude it rather than presenting an estimate as a
# measurement.
_CHARS_PER_TOKEN = 4.0


class LLMUnavailable(RuntimeError):
    """Raised when the model could not be reached or returned nothing usable."""


def estimate_tokens(text: str) -> int:
    """Approximate token count. Only used when the provider omits `usage`."""
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def token_stats(messages: Sequence[dict[str, Any]], completion: str) -> tuple[int, int]:
    """(prompt, completion) token counts, estimated from text.

    Separated from `complete()` so a caller holding a rendered prompt -- for
    example to decide whether the context budget is worth its cost -- can price
    it without making a network call.
    """
    prompt_text = " ".join(str(message.get("content", "")) for message in messages)
    return estimate_tokens(prompt_text), estimate_tokens(completion)


def approx_cost_usd(settings: Settings, usage: Usage) -> float:
    """Cost estimate in USD from configurable per-1M-token list prices.

    An *estimate* against published list prices, not billing. Real spend depends
    on the negotiated rate, cached-token discounts and the fallback model's
    price. The rates are configurable because they drift; see
    `docs/observability.md` for the provenance caveat.
    """
    return (
        usage.prompt_tokens / 1_000_000 * settings.cost_per_mtok_input
        + usage.completion_tokens / 1_000_000 * settings.cost_per_mtok_output
    )


def _post(
    settings: Settings,
    messages: Sequence[dict[str, Any]],
    *,
    temperature: float,
    max_tokens: int | None,
) -> dict[str, Any]:
    """One chat completion with bounded retries. Returns the decoded payload."""
    body: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": list(messages),
        "temperature": temperature,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens

    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"

    url = f"{settings.llm_base_url}/chat/completions"
    attempts = max(1, settings.llm_max_retries + 1)
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(body),
                timeout=settings.llm_timeout_s,
            )
            last_attempt = attempt == attempts - 1
            if response.status_code in _RETRY_STATUSES and not last_attempt:
                # Full jitter rather than fixed backoff: synchronised retries
                # against a struggling provider are how a brownout becomes an
                # outage.
                time.sleep(random.uniform(0, 2**attempt))
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= attempts - 1:
                break
            time.sleep(random.uniform(0, 2**attempt))

    raise LLMUnavailable(
        f"generation request to {settings.llm_model} failed after {attempts} attempt(s): {last_error}"
    )


def _extract(payload: dict[str, Any], prompt_tokens_hint: int) -> tuple[str, Usage]:
    """Pull the completion text and usage out of a provider response."""
    try:
        content = payload["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMUnavailable(f"malformed generation response: {exc}") from exc

    raw = payload.get("usage") or {}
    prompt_tokens = raw.get("prompt_tokens")
    completion_tokens = raw.get("completion_tokens")

    if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        usage = Usage(
            model=raw.get("model") or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=0.0,
            estimated=False,
        )
    else:
        # No usage block. Estimate from the text we already hold and mark it, so
        # a cost report can separate measured spend from inferred spend instead
        # of quietly conflating them.
        usage = Usage(
            model="",
            prompt_tokens=prompt_tokens_hint or estimate_tokens(content),
            completion_tokens=estimate_tokens(content),
            latency_ms=0.0,
            estimated=True,
        )
    return content, usage


def complete(
    settings: Settings,
    messages: Sequence[dict[str, Any]],
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> tuple[str, Usage]:
    """Single chat completion. Returns `(content, usage)`.

    Raises `LLMUnavailable` when no provider is configured or the call fails.
    Callers decide whether that is fatal; the pipeline degrades to the
    extractive generator rather than returning a 500, because an unconfigured
    optional dependency is a normal state for this project, not an incident.
    """
    if not settings.llm_enabled:
        raise LLMUnavailable(
            "no generation provider configured: set LLM_API_KEY (or point "
            "LLM_BASE_URL at a local server and leave the key unset)"
        )

    started = time.perf_counter()
    payload = _post(settings, messages, temperature=temperature, max_tokens=max_tokens)
    prompt_hint, _ = token_stats(messages, "")
    content, usage = _extract(payload, prompt_hint)

    usage = Usage(
        model=usage.model or settings.llm_model,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        cost_usd=approx_cost_usd(settings, usage),
        estimated=usage.estimated,
        cached=False,
    )
    return content, usage
