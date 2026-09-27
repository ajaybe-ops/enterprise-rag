"""Rank fusion: combine sparse and dense candidate lists into one.

Two strategies, and having both is instructive.

**RRF (Reciprocal Rank Fusion)** — Cormack et al. 2009. Each document scores
`Σ 1 / (k + rank_i(d))` across lists, k defaulting to 60.

The appeal is that it consumes *only* ranks, never raw scores. That sidesteps a
real problem: BM25 scores are unbounded and corpus-dependent, cosine
similarities are bounded but a different kind of quantity, and the two cannot be
meaningfully added without calibration. RRF needs no normalisation and no score
scale agreement. The `k` constant flattens the contribution of top ranks — with
k=60 the difference between rank 1 and rank 2 is about 3%, so a single list
cannot dominate the fusion by stacking its own best hits.

**Weighted score fusion** — min-max normalise each list per query, then take a
weighted sum. This *does* use score magnitudes, so a document that is
overwhelmingly the best match for one retriever can outrank one that merely
appeared high in both. More expressive, and it depends on the normalisation
being right and the weights being tuned — which is what the benchmark sweep is
for.

Default is RRF, for the score-scale reason above rather than because it is the
popular choice.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.types import Chunk

__all__ = ["rrf_fuse", "weighted_fuse", "fuse"]

# Per-retriever list order is fixed so `dense_weight`/`sparse_weight` have an
# unambiguous referent. Sparse-first, because with a single list enabled the
# remaining weight should be the dense one.
DENSE_ATTR = "dense_score"
SPARSE_ATTR = "bm25_score"


def _normalise(values: Sequence[float]) -> list[float]:
    """Min-max to [0, 1].

    A list of identical values maps to all-1.0 rather than dividing by zero: a
    flat score list means "this retriever found no discrimination", not "every
    document is maximally relevant".
    """
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return [1.0] * len(values)
    span = high - low
    return [(v - low) / span for v in values]


def rrf_fuse(
    ranked_lists: Sequence[Sequence[Chunk]],
    weights: Sequence[float] | None = None,
    *,
    rrf_k: int = 60,
    final_k: int = 10,
) -> list[Chunk]:
    """Fuse by reciprocal rank. Each input list is in descending quality order."""
    totals: dict[str, float] = {}
    best: dict[str, Chunk] = {}
    # (which list surfaced it, position within that list) — the tie-break, so
    # fusion is deterministic run to run.
    provenance: dict[str, tuple[int, int]] = {}

    for list_index, ranked in enumerate(ranked_lists):
        weight = weights[list_index] if weights and list_index < len(weights) else 1.0
        for position, chunk in enumerate(ranked):
            # 1-based rank. A 0-based `k + position` would make the top hit the
            # single strongest possible contribution in the whole fusion.
            totals[chunk.chunk_id] = totals.get(chunk.chunk_id, 0.0) + weight / (rrf_k + position + 1)
            if chunk.chunk_id not in best:
                best[chunk.chunk_id] = chunk
                provenance[chunk.chunk_id] = (list_index, position)

    order = sorted(
        totals.items(),
        key=lambda item: (-item[1], provenance[item[0]], item[0]),
    )[:final_k]

    return [
        best[chunk_id].with_scores(rank=rank, fused_score=score)
        for rank, (chunk_id, score) in enumerate(order)
    ]


def weighted_fuse(
    ranked_lists: Sequence[Sequence[Chunk]],
    attrs: Sequence[str],
    weights: Sequence[float] | None = None,
    *,
    final_k: int = 10,
) -> list[Chunk]:
    """Fuse by min-max-normalised, weighted score sum.

    `attrs[i]` names the score field read from `ranked_lists[i]`, so a sparse
    list can contribute `bm25_score` while a dense list contributes
    `dense_score` in the same pass. Doing this in one accumulator matters:
    summing two independent passes double-counts every document both retrievers
    returned, which silently inflates agreement.
    """
    if attrs and len(attrs) != len(ranked_lists):
        raise ValueError(
            f"attrs has {len(attrs)} entries but there are {len(ranked_lists)} ranked lists"
        )

    totals: dict[str, float] = {}
    best: dict[str, Chunk] = {}
    order_seen: dict[str, int] = {}
    counter = 0

    for list_index, ranked in enumerate(ranked_lists):
        weight = weights[list_index] if weights and list_index < len(weights) else 1.0
        attr = attrs[list_index] if attrs else DENSE_ATTR
        filled = [float(getattr(c, attr, 0.0) or 0.0) for c in ranked]
        for chunk, value in zip(ranked, _normalise(filled)):
            totals[chunk.chunk_id] = totals.get(chunk.chunk_id, 0.0) + weight * value
            if chunk.chunk_id not in best:
                best[chunk.chunk_id] = chunk
                order_seen[chunk.chunk_id] = counter
                counter += 1

    order = sorted(totals.items(), key=lambda item: (-item[1], order_seen[item[0]]))[:final_k]
    return [
        best[chunk_id].with_scores(rank=rank, fused_score=score)
        for rank, (chunk_id, score) in enumerate(order)
    ]


def fuse(
    ranked_lists: Sequence[Sequence[Chunk]],
    strategy: str = "rrf",
    *,
    final_k: int = 10,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> list[Chunk]:
    """Dispatch fusion over [dense, sparse] lists.

    List order is fixed: index 0 is dense, index 1 is sparse. That makes the
    weights unambiguous and keeps the benchmark's comparisons honest — a sweep
    that reorders lists between runs is not measuring the weights.
    """
    lists = [list(r) for r in ranked_lists if r]
    if not lists:
        return []

    if len(lists) == 1:
        weights = [dense_weight if dense_weight > 0 else sparse_weight]
        attrs = [DENSE_ATTR]
    else:
        weights = [dense_weight, sparse_weight]
        attrs = [DENSE_ATTR, SPARSE_ATTR]

    if strategy == "rrf":
        return rrf_fuse(lists, weights, rrf_k=rrf_k, final_k=final_k)
    if strategy == "weighted":
        return weighted_fuse(lists, attrs, weights, final_k=final_k)
    raise ValueError(f"unknown fusion strategy {strategy!r}; expected 'rrf' or 'weighted'")
