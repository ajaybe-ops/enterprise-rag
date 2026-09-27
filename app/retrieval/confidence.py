"""Confidence estimation and adaptive retrieval.

The contribution
----------------
Everything else in this repo is standard practice. This module is the part that
is a design decision, so it is worth stating the problem precisely.

A single retrieval pass is a fixed bet: take `final_k` chunks, hand them to the
model, done. But the quality of that bet varies enormously across queries, and
nothing in a conventional pipeline knows which bet it made. Worse, the failure is
*asymmetric*:

* **False confidence** — the retrieved chunks look plausible but none of them
  answers the question. The model, having been handed five irrelevant passages
  and told to answer from them, will either hallucinate or produce a
  confidently-shaped non-answer. This is the expensive failure.
* **Honest abstention** — the pipeline recognises thin evidence and declines.
  The user gets a worse answer than a lucky retrieval would have given, and
  costs nothing but the abstention itself.

So the pipeline should measure how good its own evidence is, and spend more when
it is bad. That is what this does.

The confidence signal
---------------------
Three signals, each chosen because it fails in a *different* way, so their
disagreement is itself informative:

1. **`top_score`** — the reranker's score for the best chunk. Absolute quality
   of the single best piece of evidence. Blind to whether the *rest* of the
   evidence is related.

2. **`margin`** — `(top - second) / top`. A high margin means one chunk clearly
   dominates. A margin near zero means five chunks are equally mediocre, which
   is the signature of a query the retriever did not understand. This is the
   signal that catches "relevant-looking but wrong", which `top_score` misses
   whenever the reranker is over-confident.

3. **`agreement`** — overlap between the sparse and dense rankings. When both
   retrievers independently surface the same documents, the evidence is
   corroborated by two different failure modes. When they disagree completely,
   at least one of them is operating outside its competence on this query, and
   that is worth knowing before trusting the union.

4. **`coverage`** — what fraction of the query's content words actually appear
   in the evidence. The first three signals are all *relative*: they compare
   chunks against each other, so a query with nothing to do with the corpus
   still produces a ranking, still produces scores, and still looks internally
   consistent. An out-of-domain query is not a low-quality ranking, it is a
   ranking of the wrong thing, and no amount of comparing it to itself reveals
   that. Coverage is the only signal that looks at the *query* rather than at
   the results, which is why it catches the case the other three structurally
   cannot.

Fusing them multiplicatively rather than by averaging is deliberate. Confidence
that one signal is high should not paper over another being low: an average
lets a strong `top_score` compensate for `margin ≈ 0`, which is exactly the
false-confidence case this is trying to prevent. A product stays low if any
factor is low, so a query that is topically fine but ambiguous overall still
escalates.

Escalation
----------
Below `confidence_threshold`, retry once with a wider candidate pool and, if the
query transformer produced variants, those variants too. Below `abstain_below`,
refuse to answer.

Why only one escalation: each escalation is a second embedding call and a second
rerank, so cost roughly doubles. The benchmark measures the recall gained per
escalation against that cost, and the default `max_escalations=1` is where the
measured trade-off stops paying. Raising it is a tuning decision backed by
numbers in docs/09_EXPERIMENTS.md, not a preference.

Honesty about the limits
------------------------
`top_score` from a cross-encoder is *not* a calibrated probability. A reranker
score of 0.9 on one query is not comparable to 0.9 on another, and the
absolute scale differs per model. The thresholds are therefore **operating
points chosen against a labelled query set**, not learned probabilities, and
they are exposed in `RetrievalConfig` precisely so the choice is visible and
revisable. A deployment that needs real calibration should fit the thresholds on
its own data; `calibration_curve()` returns the raw material for that.

What has actually been measured here
------------------------------------
`scripts/probe_confidence_separation.py` runs four in-domain and four
off-domain questions against the 16-chunk corpus. With `coverage` included, the
in-domain scores span 0.26-0.39 and the off-domain span 0.07-0.25, so the two
groups separate — but the margin between the worst in-domain question and the
best off-domain one is about **0.01**, on a sample of eight hand-picked
questions.

That is enough to say the signals are not decorative, and nowhere near enough
to call any threshold validated. Treat the abstain floor as a provisional
operating point, not a measured result. A real calibration needs a labelled set
of in-domain and out-of-domain questions, which this repository does not yet
have; that gap is the reason `docs/09_EXPERIMENTS.md` cannot currently claim
an abstention quality figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.text import analyze
from app.types import Chunk, RetrievalResult

__all__ = ["ConfidenceSignals", "estimate_confidence", "should_escalate", "should_abstain", "calibration_curve"]

# Weight each signal's shortfall by how much it should be able to veto. The
# strongest veto goes to margin, because margin is the signal that catches the
# expensive failure mode.
_SIGNAILS = {
    "top_score": 0.9,   # maps [0, 1] -> [veto_floor, 1]
    "margin": 0.85,
    "agreement": 0.75,
    # The strongest veto: coverage is the only signal that can say "this
    # evidence is about something else entirely", and on a 16-chunk corpus an
    # off-domain query otherwise scores mid-0.4s because the reranker finds
    # *something* plausible in every document.
    "coverage": 0.95,
}
_VETO_FLOOR = 0.35  # a signal at 0 pulls confidence to 0.35, not to 0

# A query shorter than this in content words is too small for coverage to mean
# anything: "what is X" is one content word, and missing it is not evidence of
# an off-domain query, it is evidence of a terse one.
_MIN_COVERAGE_TERMS = 2

# Below this coverage the evidence is not about the query, rather than being
# weakly about it. A linear weight let `coverage=0.33` cost only a bounded
# amount, which was not enough: measured against a hand-picked probe set, a
# "how do I train for a marathon" query still outscored a genuine in-domain
# question. Quadratic falloff below the floor encodes the real shape of the
# signal — absent content words are not a partial deduction.
_COVERAGE_FLOOR = 0.5


def _sharpen_coverage(coverage: float) -> float:
    """Quadratic falloff below `_COVERAGE_FLOOR`, identity above it.

    At the floor the two branches agree, so a well-covered query is untouched
    and only the ambiguous low range is pushed down.
    """
    if coverage >= _COVERAGE_FLOOR:
        return coverage
    return coverage * (coverage / _COVERAGE_FLOOR)


@dataclass(frozen=True, slots=True)
class ConfidenceSignals:
    top_score: float
    margin: float
    agreement: float
    coverage: float
    agreement_detail: dict[str, float]

    @property
    def confidence(self) -> float:
        """Multiplicative fusion of the four signals.

        `v = 1 - (1 - s_i) * weight_i` maps a signal in [0,1] to
        [VETO_FLOOR, 1], so a zero signal costs a bounded amount and the product
        of the bounded factors stays interpretable: each is "how close to fully
        trusting this signal are we".
        """
        product = 1.0
        for value, weight in (
            (self.top_score, _SIGNAILS["top_score"]),
            (self.margin, _SIGNAILS["margin"]),
            (self.agreement, _SIGNAILS["agreement"]),
            (_sharpen_coverage(self.coverage), _SIGNAILS["coverage"]),
        ):
            clamped = min(1.0, max(0.0, value))
            product *= 1.0 - (1.0 - clamped) * weight * (1.0 - _VETO_FLOOR)
        return round(min(1.0, product), 4)

    def to_dict(self) -> dict[str, float]:
        return {
            "top_score": round(self.top_score, 4),
            "margin": round(self.margin, 4),
            "agreement": round(self.agreement, 4),
            "coverage": round(self.coverage, 4),
            "confidence": self.confidence,
        }


def _rank_overlap(a: Sequence[Chunk], b: Sequence[Chunk], depth: int = 10) -> float:
    """Mean reciprocal-rank-ish overlap of two rankings.

    Not Jaccard: Jaccard on the *set* throws away ordering, which is the thing
    being compared. Using RRF over the two lists keeps the ordering signal, and
    normalising by the list length makes it comparable across queries whose
    result sets differ in size.
    """
    if not a or not b:
        return 0.0
    depth = min(depth, len(a), len(b))
    rank_a = {c.chunk_id: i + 1 for i, c in enumerate(a[:depth])}
    rank_b = {c.chunk_id: i + 1 for i, c in enumerate(b[:depth])}
    shared = set(rank_a) & set(rank_b)
    if not shared:
        return 0.0
    rrf_k = 60
    score = sum(1.0 / (rrf_k + rank_a[cid]) + 1.0 / (rrf_k + rank_b[cid]) for cid in shared)
    # Normalise by the score a perfect overlap would produce, i.e. every member
    # of the smaller list agreeing at identical ranks.
    ceiling = sum(1.0 / (rrf_k + r) + 1.0 / (rrf_k + r) for r in range(1, depth + 1))
    return min(1.0, score / ceiling) if ceiling else 0.0


def _query_coverage(query: str, reranked: Sequence[Chunk], depth: int = 5) -> float:
    """Fraction of the query's content words present in the best evidence.

    Best-of-`depth` rather than pooled across all evidence: a corpus can
    contain every word of a question in five different documents and still not
    answer it, so pooling would score topical adjacency as coverage. Scoring the
    best single chunk asks the question that actually matters — "is there one
    piece of evidence this query is about?".

    Terms are stemmed so `restarting` matches `restart`, and duplicates are
    collapsed so a repeated word is not double-weighted.
    """
    if not query or not reranked:
        return 1.0
    terms = set(analyze(query))
    if len(terms) < _MIN_COVERAGE_TERMS:
        # Too few content words for absence to mean anything.
        return 1.0

    best = 0.0
    for chunk in reranked[:depth]:
        chunk_terms = set(analyze(chunk.text))
        if not chunk_terms:
            continue
        best = max(best, len(terms & chunk_terms) / len(terms))
        if best >= 1.0:
            break
    return best


def estimate_confidence(
    reranked: Sequence[Chunk],
    sparse_ranking: Sequence[Chunk] | None = None,
    dense_ranking: Sequence[Chunk] | None = None,
    query: str = "",
) -> ConfidenceSignals:
    """Compute the four confidence signals from one pass's outputs.

    Signals that cannot be computed degrade to neutral (1.0) rather than zero.
    Running sparse-only means there is no disagreement to measure, and scoring
    that as "no agreement" would make every single-retriever query look
    maximally uncertain and escalate forever. Coverage likewise degrades to
    neutral when no query text is available to measure.
    """
    if not reranked:
        return ConfidenceSignals(0.0, 0.0, 1.0, 1.0, {})

    scores = [c.rerank_score for c in reranked if c.rerank_score is not None]
    if scores:
        top = max(scores)
        # A negative or zero top score is a reranker telling us nothing matched.
        top_signal = top if top > 0 else 0.0
        if len(scores) > 1:
            ordered = sorted(scores, reverse=True)
            second = ordered[1]
            margin = (ordered[0] - second) / ordered[0] if ordered[0] > 0 else 0.0
        else:
            # A single candidate cannot be distinguished from the void. Cap
            # rather than default to 1.0 — one chunk is not strong evidence.
            margin = 0.5
    else:
        top_signal, margin = 0.0, 0.0

    if sparse_ranking and dense_ranking:
        agreement = _rank_overlap(sparse_ranking, dense_ranking)
        detail = {
            "sparse_returned": float(len(sparse_ranking)),
            "dense_returned": float(len(dense_ranking)),
            "overlap_at_10": round(agreement, 4),
        }
    else:
        agreement = 1.0
        detail = {"note": 0.0} if not (sparse_ranking and dense_ranking) else {}

    coverage = _query_coverage(query, reranked) if query else 1.0

    return ConfidenceSignals(
        top_score=min(1.0, max(0.0, top_signal)),
        margin=min(1.0, max(0.0, margin)),
        agreement=agreement,
        coverage=min(1.0, max(0.0, coverage)),
        agreement_detail=detail,
    )


def should_escalate(signals: ConfidenceSignals, threshold: float, escalations: int, max_escalations: int) -> bool:
    return escalations < max_escalations and signals.confidence < threshold


def should_abstain(signals: ConfidenceSignals, floor: float) -> bool:
    return signals.confidence < floor


def calibration_curve(
    results: Sequence[RetrievalResult],
    buckets: int = 10,
) -> list[dict[str, float]]:
    """Bin results by predicted confidence and report mean confidence per bin.

    This is the honest way to present a confidence estimate: it does not claim
    the estimate is calibrated, it lets you *see* whether it is. A well-behaved
    estimator has mean-confidence rising monotonically across bins; a flat or
    decreasing curve means the signals are not separating good retrievals from
    bad ones and the thresholds are decorative.
    """
    if not results:
        return []
    edges = [i / buckets for i in range(buckets + 1)]
    out: list[dict[str, float]] = []
    for low, high in zip(edges, edges[1:]):
        members = [r for r in results if low <= r.confidence < high or (high == 1.0 and r.confidence == 1.0)]
        if not members:
            continue
        out.append(
            {
                "bin_low": round(low, 3),
                "bin_high": round(high, 3),
                "count": float(len(members)),
                "mean_confidence": round(sum(r.confidence for r in members) / len(members), 4),
                "mean_chunks": round(sum(len(r.chunks) for r in members) / len(members), 3),
            }
        )
    return out
