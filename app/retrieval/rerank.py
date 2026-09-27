"""Reranking: re-score a candidate list with a cross-encoder.

Why a second stage
------------------
A bi-encoder embeds the query and the document independently and compares the
resulting vectors. That independence is the weakness: word order, negation and
specificity are all lost in the embedding, and the model cannot attend to
"this document answers *that* question" as a pair.

A cross-encoder reads the query and the document together and scores the pair.
Substantially more accurate, and far too slow to run against the whole corpus —
so it runs on the top-`candidate_k` only, and the pipeline pays for the
precision on a bounded workload.

Three implementations, chosen by what is available:

* `jina` — production. `jina-reranker-v2-base-multilingual` over the API.
* `lexical_overlap` — offline. IDF-weighted query-term coverage. Genuinely
  weaker than a cross-encoder, and reported as such: the benchmark labels which
  reranker produced a number so its results are never read as the neural
  reranker's results.
* `none` — pass-through, still re-ranked and re-scored so downstream
  confidence maths has a defined input.

The invariant every implementation upholds: **the returned list is in reranked
order, and each `Chunk` carries its own `rerank_score` and provenance.** The
previous implementation returned bare strings and the caller re-derived order
from a parallel structure, which discarded the ranking entirely — see
docs/AUDIT.md finding 1.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from typing import Sequence

from app.config import Settings
from app.text import analyze
from app.types import Chunk, renumber

__all__ = ["rerank", "lexical_overlap_scores"]


def lexical_overlap_scores(query: str, chunks: Sequence[Chunk]) -> list[float]:
    """IDF-weighted query-term coverage, in [0, 1].

    A stand-in for a cross-encoder when no reranking API is configured. It
    rewards chunks that cover the *distinctive* query terms (rare terms carry
    more IDF) and normalises by the number of query terms, so it is comparable
    across queries.
    """
    query_terms = set(analyze(query))
    if not query_terms or not chunks:
        return [0.0] * len(chunks)

    # IDF estimated from the candidate pool itself. Small pool, so these are
    # noisy estimates — adequate for ranking, not for reporting.
    doc_freq: Counter = Counter()
    per_doc: list[set[str]] = []
    for chunk in chunks:
        terms = set(analyze(chunk.text))
        per_doc.append(terms)
        doc_freq.update(terms)

    n = len(chunks)
    idf = {
        term: math.log(1.0 + (n - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
        for term in query_terms
    }
    total_idf = sum(idf.values()) or 1.0

    scores: list[float] = []
    for terms in per_doc:
        covered = sum(idf[t] for t in query_terms if t in terms)
        scores.append(covered / total_idf)
    return scores


def _jina_rerank(
    settings: Settings, query: str, chunks: Sequence[Chunk], top_n: int
) -> list[Chunk] | None:
    import requests

    payload = {
        "model": settings.jina_rerank_model,
        "query": query,
        "documents": [c.text for c in chunks],
        "top_n": min(top_n, len(chunks)),
    }
    response = requests.post(
        f"{settings.jina_base_url}/rerank",
        headers={
            "Authorization": f"Bearer {settings.jina_api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=settings.jina_timeout_s,
    )
    response.raise_for_status()
    results = response.json()["results"]

    ranked: list[Chunk] = []
    for item in results:
        index = item.get("index")
        # An out-of-range index is a malformed response, not a missing document.
        # Indexing blindly would attach the wrong score to the wrong chunk.
        if not isinstance(index, int) or not 0 <= index < len(chunks):
            continue
        ranked.append(
            chunks[index].with_scores(rerank_score=float(item.get("relevance_score", 0.0)))
        )
    return ranked or None


def rerank(
    settings: Settings,
    query: str,
    chunks: Sequence[Chunk],
    top_n: int,
) -> tuple[list[Chunk], str, float]:
    """Rerank `chunks` against `query`, returning (top chunks, strategy, ms).

    The order of the returned list is the reranked order. `strategy` names the
    implementation that actually ran, which may differ from the configured one
    (a configured `jina` reranker with no API key degrades and says so).
    """
    if not chunks:
        return [], "none", 0.0

    started = time.perf_counter()
    requested = settings.retrieval.rerank

    if requested == "jina" and settings.rerank_enabled:
        try:
            ranked = _jina_rerank(settings, query, chunks, top_n)
            if ranked:
                return renumber(ranked[:top_n]), "jina", _ms(started)
        except Exception:  # noqa: BLE001 - degrade, never fail the request
            # Fall through to the offline reranker rather than returning
            # unrereanked candidates under a "jina" label.
            pass

    if requested == "none":
        # Still assign descending scores from the incoming order so the
        # confidence calculation has a defined, monotonic input.
        scored = [
            chunk.with_scores(
                rerank_score=(1.0 - i / max(1, len(chunks) - 1)) if len(chunks) > 1 else 1.0
            )
            for i, chunk in enumerate(chunks)
        ]
        return renumber(scored[:top_n]), "none", _ms(started)

    scores = lexical_overlap_scores(query, chunks)
    ordered = sorted(
        zip(chunks, scores), key=lambda pair: -pair[1]
    )[:top_n]
    return (
        renumber([c.with_scores(rerank_score=s) for c, s in ordered]),
        "lexical_overlap",
        _ms(started),
    )


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0
