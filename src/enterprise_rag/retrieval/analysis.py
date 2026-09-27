"""Query transformation.

Two mechanisms, both of which degrade to the original question when no LLM is
configured, because a query transformer that only works with a key is a
dependency, not a feature.

**Rewrite.** The user's phrasing and the document's phrasing differ.
"How do I see if my batch thing finished" needs to reach a document that says
"monitor the status of a Kubernetes Job". A rewrite closes that gap. With no
LLM, the fallback is a domain-synonym expansion table, applied as token
substitution. That is weaker than an LLM rewrite and is not claimed to be
equivalent — it exists so the offline path is not *zero* help.

**Multi-query.** Rather than betting everything on one phrasing, generate
several and retrieve each, then fuse. The bet is that if one phrasing misses,
another lands. This is the classic trade: recall goes up, cost goes up
linearly in the number of variants, and the marginal variant is worth less than
the first. That is why `variants` defaults to 2, not 5.

**Extracted entities** (`kubectl`, `HorizontalPodAutoscaler`, `--field-selector`)
are pulled out separately and appended as a high-precision lexical variant.
Exact identifiers are precisely what dense retrieval handles worst and BM25
handles perfectly, so an identifier-only variant is a cheap, targeted probe
against the sparse side of the hybrid.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from enterprise_rag.config.settings import Settings
from enterprise_rag.text import analyze
from enterprise_rag.types import QueryAnalysis

__all__ = ["analyse_query", "expand_with_synonyms", "extract_entities"]

# Curated substitutions. Each maps a user-ism to the vocabulary the corpus
# actually uses. Kept small and hand-checked against the corpus: an unvalidated
# synonym table silently damages precision on queries it gets wrong, and a large
# auto-generated one is worse than none.
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "pod": ("container",),
    "pods": ("container",),
    "job": ("batch",),
    "jobs": ("batch",),
    "batch": ("job",),
    "crashloopbackoff": ("crash", "restart"),
    "scale": ("autoscal",),
    "scaling": ("autoscal", "hpa"),
    "autoscale": ("hpa", "autoscal"),
    "network": ("cni", "sdn"),
    "cpu": ("resource", "limit"),
    "memory": ("resource", "limit"),
    "restart": ("backoff", "crash"),
    "status": ("state", "condition"),
    "monitor": ("observe", "metric"),
    "monitoring": ("observe", "metric"),
    "log": ("logs",),
    "logs": ("log",),
}

# Identifiers worth probing lexically. Flags, resource kinds, dotted API paths.
_IDENTIFIER_RE = re.compile(
    r"""(?x)
    (?:--[a-z][a-z0-9-]+)          # --field-selector
    | (?:\b[a-z]+\.[a-z0-9.]+\b)   # enterprise_rag.kubernetes.io
    | (?:\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b)  # HorizontalPodAutoscaler
    | (?:\bkubectl\s+[a-z]+\b)     # kubectl get
    | (?:\b(?:minikube|etcd|kubelet|sidecar
              |StatefulSet|DaemonSet|CronJob
              |Deployment|Service|Ingress
              |PodDisruptionBudget
              |HorizontalPodAutoscaler|VerticalPodAutoscaler)\b)
    """
)


def extract_entities(text: str) -> tuple[str, ...]:
    """Exact identifiers, deduplicated, order preserved."""
    seen: dict[str, None] = {}
    for match in _IDENTIFIER_RE.findall(text):
        token = match.strip()
        if token:
            seen.setdefault(token, None)
    return tuple(seen)


def expand_with_synonyms(text: str) -> str:
    """Append corpus vocabulary for recognised user-isms.

    Additive rather than substitutive: removing the user's own words would break
    the lexical probe for the very terms the user supplied.
    """
    tokens = analyze(text, stem_tokens=False)
    additions: list[str] = []
    for token in tokens:
        for synonym in _SYNONYMS.get(token, ()):
            if synonym not in additions:
                additions.append(synonym)
    return f"{text} {' '.join(additions)}".strip() if additions else text


def _llm_rewrite(settings: Settings, question: str, history: str) -> str | None:
    """One small LLM call to rewrite the question. None on any failure."""
    from enterprise_rag.generation.provider import complete

    prompt = (
        "Rewrite the user's question as a single search query for a technical "
        "document index about Kubernetes, Intel hardware and enterprise "
        "networking. Keep identifiers (flags, resource kinds, API paths) "
        "verbatim. Remove conversational filler. Output only the query.\n\n"
        f"{('Conversation so far: ' + history + chr(10)) if history else ''}"
        f"Question: {question}\nQuery:"
    )
    try:
        content, _usage = complete(
            settings,
            messages=[{"role": "user", "content": prompt}],
            feature="query_rewrite",
        )
    except Exception:
        return None
    cleaned = (content or "").strip().strip('"').splitlines()[0] if content else ""
    return cleaned[:300] or None


def analyse_query(
    settings: Settings,
    question: str,
    history: Sequence[dict] | None = None,
    *,
    conversational_hint: bool = False,
) -> QueryAnalysis:
    """Transform a user question into one or more retrieval queries.

    Always returns something usable. `method` records which path produced it so
    the benchmark can separate the effect of the transformer from the effect of
    the retriever.
    """
    question = question.strip()
    entities = extract_entities(question)

    if settings.llm_enabled:
        history_text = ""
        if history:
            history_text = " ".join(
                str(m.get("content", ""))[-300:] for m in history[-4:] if isinstance(m, dict)
            )
        rewritten = _llm_rewrite(settings, question, history_text)
        if rewritten:
            variants: list[str] = [expand_with_synonyms(question)]
            if entities:
                variants.append(" ".join(entities))
            return QueryAnalysis(
                original=question,
                rewritten=rewritten,
                variants=tuple(v for v in variants if v and v != question),
                is_conversational=conversational_hint,
                entities=entities,
                method="llm_rewrite",
            )

    # Offline path: lexical expansion only. Honest about being weaker.
    rewritten = expand_with_synonyms(question)
    variants = []
    if entities:
        variants.append(" ".join(entities))
    return QueryAnalysis(
        original=question,
        rewritten=rewritten,
        variants=tuple(variants),
        is_conversational=conversational_hint,
        entities=entities,
        method="lexical_rewrite",
    )
