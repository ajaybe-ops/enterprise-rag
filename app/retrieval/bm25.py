"""BM25 lexical retrieval.

Why this is here at all
-----------------------
The corpus is Kubernetes and networking documentation, which is dense with exact
identifiers: `kubectl get jobs`, `--field-selector`, `HorizontalPodAutoscaler`,
`spec.completions`. A bi-encoder embeds those into a vector where the *shape* of
the token matters more than the token itself, so `CronJob` and `cron job` land
close together while `statefulset` and `stateful` land much closer than either
deserves. Dense retrieval is weakest precisely where this corpus is strongest.

BM25 is not a fuzzy competitor to dense retrieval, it is a different failure
mode: it is exact, lexical, and interpretable. Its weakness is vocabulary
mismatch ("pod restarts" vs "container restart policy"), which is exactly dense
retrieval's strength. Fusing them is the point.

The implementation
------------------
Standard Robertson/Sparck-Jones IDF with the usual epsilon smoothing:

    IDF(t) = ln( 1 + (N - n_t + 0.5) / (n_t + 0.5) )

    score(q, d) = Σ_{t∈q}  IDF(t) · ( f(t,d)·(k1+1) )
                                 ─────────────────────────────────
                                 f(t,d) + k1·(1 - b + b·|d|/avgdl)

Smoothing matters. The unsmoothed `ln((N - n_t + 0.5)/(n_t + 0.5))` goes negative
for terms appearing in more than half the documents, so a *very* common term
actively subtracts from the score. The `1 +` keeps IDF non-negative, which stops
a corpus-wide term like "kubernetes" from penalising every document that
contains it.

Term-frequency saturation (`k1`) and length normalisation (`b`) are the two
knobs. They are exposed in `RetrievalConfig` so the benchmark can sweep them
rather than taking them on faith.

No dependency: `rank_bm25` is a thin, unmaintained wrapper around this same
formula. Having it in-tree means the scoring is unit-testable against
hand-computed values, which is the only way to be confident a change to it is
not a change to recall.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from app.text import analyze
from app.types import Chunk

__all__ = ["BM25Index", "bm25_scores"]


@dataclass(frozen=True, slots=True)
class BM25Index:
    """Immutable inverted index over a chunk collection.

    Built once per corpus. `scores()` is allocation-light for a small corpus and
    correctness is the priority; a production deployment would hold postings
    as arrays, which is noted in docs/adr/0004.
    """

    chunks: tuple[Chunk, ...]
    k1: float
    b: float

    # Derived state, computed in __post_init__ via object.__setattr__ (frozen
    # dataclass). These MUST have defaults: a dataclass field without one becomes
    # a required __init__ argument, so declaring them bare made
    # `BM25Index(chunks=..., k1=..., b=...)` fail with "missing 5 required
    # positional arguments" and __post_init__ unreachable. Sparse retrieval was
    # therefore not merely untested — it was unconstructible.
    _term_freqs: tuple[Counter, ...] = ()
    _doc_freqs: dict[str, int] = field(default_factory=dict)
    _doc_lens: tuple[int, ...] = ()
    _avgdl: float = 0.0
    _idf: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n_docs = len(self.chunks)
        term_freqs: list[Counter] = []
        doc_freqs: Counter = Counter()
        doc_lens: list[int] = []

        for chunk in self.chunks:
            tokens = analyze(chunk.text)
            freqs = Counter(tokens)
            term_freqs.append(freqs)
            doc_freqs.update(freqs.keys())
            doc_lens.append(len(tokens))

        avgdl = (sum(doc_lens) / n_docs) if n_docs else 0.0
        idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freqs.items()
        }

        object.__setattr__(self, "_term_freqs", tuple(term_freqs))
        object.__setattr__(self, "_doc_freqs", dict(doc_freqs))
        object.__setattr__(self, "_doc_lens", tuple(doc_lens))
        object.__setattr__(self, "_avgdl", avgdl)
        object.__setattr__(self, "_idf", idf)

    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def vocabulary_size(self) -> int:
        return len(self._doc_freqs)

    def idf(self, term: str) -> float:
        """IDF for a term, including out-of-vocabulary terms.

        An OOV term gets the IDF of a term seen exactly once: high, but finite.
        Returning 0 would silently drop the term and make an unseen keyword look
        like a non-match, which is a recall bug that is very hard to see.
        """
        return self._idf.get(term, math.log(1.0 + (len(self.chunks) + 0.5) / 1.5))

    def scores(self, query: str) -> list[float]:
        """BM25 score of every chunk against `query`, in index order."""
        if not self.chunks:
            return []
        query_terms = analyze(query)
        if not query_terms:
            return [0.0] * len(self.chunks)

        # Repeated query terms are evidence of emphasis, so weight by count.
        query_freqs = Counter(query_terms)
        k1, b, avgdl = self.k1, self.b, self._avgdl

        results: list[float] = []
        for freqs, doc_len in zip(self._term_freqs, self._doc_lens):
            score = 0.0
            # Length normalisation factor is per-document, hoisted out of the
            # term loop. In the inner loop it would be recomputed for every term.
            norm = 1.0 - b + b * (doc_len / avgdl if avgdl else 1.0)
            for term, q_count in query_freqs.items():
                tf = freqs.get(term)
                if not tf:
                    continue
                idf = self._idf.get(term)
                if idf is None:
                    continue
                score += q_count * idf * (tf * (k1 + 1.0)) / (tf + k1 * norm)
            results.append(score)

        return results

    def search(self, query: str, top_k: int) -> list[Chunk]:
        """Top-k chunks with `bm25_score` and `rank` populated, order preserved."""
        scored = self.scores(query)
        order = sorted(
            range(len(scored)),
            key=lambda i: (-scored[i], i),  # stable tie-break on index
        )[:top_k]
        return [
            self.chunks[i].with_scores(bm25_score=scored[i], rank=rank)
            for rank, i in enumerate(order)
        ]

    def explain(self, query: str, chunk_index: int) -> dict:
        """Per-term contribution for one chunk. Used by tests and `rag why`."""
        tokens = analyze(self.chunks[chunk_index].text)
        freqs = Counter(tokens)
        doc_len = self._doc_lens[chunk_index]
        norm = 1.0 - self.b + self.b * (doc_len / self._avgdl if self._avgdl else 1.0)
        per_term = {}
        for term in analyze(query):
            tf = freqs.get(term, 0)
            idf = self._idf.get(term, 0.0)
            contribution = idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm) if tf else 0.0
            per_term[term] = {
                "tf": tf,
                "idf": round(idf, 4),
                "contribution": round(contribution, 4),
            }
        return {
            "chunk_id": self.chunks[chunk_index].chunk_id,
            "source": self.chunks[chunk_index].source,
            "score": round(self.scores(query)[chunk_index], 4),
            "terms": per_term,
        }


def bm25_scores(
    query: str, documents: Sequence[str], k1: float = 1.5, b: float = 0.75
) -> list[float]:
    """Stateless BM25 over raw strings.

    Exists for tests and for scoring a candidate list that was not built by
    `BM25Index` (the reranker path). Uses the same formula, and is checked
    against `BM25Index.scores` in the test suite so they cannot drift.
    """
    corpus = [analyze(doc) for doc in documents]
    n_docs = len(corpus)
    if not n_docs:
        return []

    doc_freqs: Counter = Counter()
    term_freqs: list[Counter] = []
    doc_lens: list[int] = []
    for tokens in corpus:
        freqs = Counter(tokens)
        term_freqs.append(freqs)
        doc_freqs.update(freqs.keys())
        doc_lens.append(len(tokens))

    avgdl = sum(doc_lens) / n_docs
    idf = {
        term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
        for term, freq in doc_freqs.items()
    }

    query_freqs = Counter(analyze(query))
    out: list[float] = []
    for freqs, doc_len in zip(term_freqs, doc_lens):
        norm = 1.0 - b + b * (doc_len / avgdl if avgdl else 1.0)
        score = 0.0
        for term, q_count in query_freqs.items():
            tf = freqs.get(term)
            if not tf:
                continue
            term_idf = idf.get(term)
            if term_idf is None:
                continue
            score += q_count * term_idf * (tf * (k1 + 1.0)) / (tf + k1 * norm)
        out.append(score)
    return out


def build_index(chunks: Iterable[Chunk], k1: float = 1.5, b: float = 0.75) -> BM25Index:
    return BM25Index(chunks=tuple(chunks), k1=k1, b=b)
