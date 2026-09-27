"""Retrieval orchestration: sparse + dense -> fuse -> rerank -> confidence.

This is the module that replaced the two-file `qdrant_service` +
`ranking_service` split. The old split had a structural problem: the two halves
spoke different types (`list[dict]` in, `list[str]` out), so the caller had to
re-join them, and it re-joined them by matching text. Everything now moves
`Chunk` objects, which is what makes the rerank-ordering bug structurally
impossible rather than merely fixed.

Backends are pluggable through `Retriever` so the same pipeline runs against
Qdrant in production and against an in-memory corpus in tests and benchmarks,
with no branching in the orchestration code itself.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from enterprise_rag.config.settings import Settings
from enterprise_rag.indexing.corpus import Corpus
from enterprise_rag.reranking.rerank import rerank
from enterprise_rag.retrieval import fusion
from enterprise_rag.retrieval.analysis import analyse_query
from enterprise_rag.retrieval.confidence import (
    estimate_confidence,
    should_escalate,
)
from enterprise_rag.retrieval.dense import Embedder
from enterprise_rag.retrieval.sparse import BM25Index, build_index
from enterprise_rag.retrieval.status import StatusPolicy
from enterprise_rag.types import Chunk, QueryAnalysis, RetrievalResult, renumber

__all__ = [
    "ALLOWED_SOURCE_TYPES",
    "Corpus",
    "HybridRetriever",
    "Retriever",
    "StatusPolicy",
    "build_retrievers",
]

__all__ = ["ALLOWED_SOURCE_TYPES", "Corpus", "HybridRetriever", "Retriever", "build_retrievers"]

# The scope gate. Only these corpora are served to users; `noisy` is indexed so
# the filter has something real to exclude and so cross-corpus confusion is
# measurable rather than hypothetical.
ALLOWED_SOURCE_TYPES = frozenset({"true"})


@runtime_checkable
class Retriever(Protocol):
    """Minimal contract the pipeline depends on."""

    def search(self, query: str, top_k: int) -> list[Chunk]:
        """Return up to `top_k` chunks, best first, carrying scores."""
        ...


@dataclass
class _BM25Retriever:
    """Sparse retriever over an in-memory corpus.

    Exhaustive BM25 scoring rather than an index lookup: honest for a corpus of
    this size, and it avoids pretending to be something it is not. Qdrant (or a
    real inverted index) is the answer at scale — see docs/adr/0004.
    """

    index: BM25Index


    @property
    def kind(self) -> str:
        return "bm25"

    def search(self, query: str, top_k: int) -> list[Chunk]:
        return self.index.search(query, top_k)


class _DenseRetriever:
    def __init__(self, embedder: Embedder, chunks: tuple[Chunk, ...]) -> None:
        self._embedder = embedder
        self._chunks = chunks

    @property
    def kind(self) -> str:
        return f"dense:{self._embedder.mode}"

    def search(self, query: str, top_k: int) -> list[Chunk]:
        return self._embedder.search(query, self._chunks, top_k)


class _QdrantRetriever:
    """Dense retrieval against a Qdrant collection, with the scope filter applied
    server-side rather than in Python.

    The filter goes in the query, not after it. Post-filtering would return fewer
    than `top_k` in-scope chunks whenever the top hits happen to be out of scope
    — silently starving the reranker on exactly the queries where the corpus is
    mixed.
    """

    def __init__(self, client, collection: str, embedder: Embedder, source_types: frozenset[str]) -> None:
        self._client = client
        self._collection = collection
        self._embedder = embedder
        self._types = source_types
        self.kind = "qdrant"

    def search(self, query: str, top_k: int) -> list[Chunk]:
        from qdrant_client.http import models

        vector = self._embedder.embed_query(query).tolist()
        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(key="source_type", match=models.MatchValue(value=st))
                    for st in sorted(self._types)
                ]
            ),
            limit=top_k,
            with_payload=True,
        )
        return [
            Chunk.from_payload(point.payload or {}, vector_id=point.id, score=point.score)
            for point in response.points
        ]


def build_retrievers(
    settings: Settings,
    corpus: Corpus | None = None,
    *,
    embedder: Embedder | None = None,
) -> dict[str, Retriever]:
    """Instantiate the retrievers the configuration asks for.

    Exactly one of `corpus` (in-memory) or a configured Qdrant is required. With
    neither, an empty dict is returned and the pipeline reports an empty
    result rather than raising — a missing index is a degradation, not a crash.
    """
    cfg = settings.retrieval
    retrievers: dict[str, Retriever] = {}

    if corpus is not None:
        # The scope gate is applied here, not left to the caller. An
        # unfiltered default would make "forgot to pass allowed" indistinguishable
        # from "no scope policy", and the failure mode is serving off-domain
        # documents to a user who should never see them.
        scoped = corpus.scoped(ALLOWED_SOURCE_TYPES)
        if not scoped.chunks:
            # An empty scope is a configuration error, not an empty index: it
            # means every document was filtered out, and serving zero results
            # without saying so looks identical to a corpus that failed to load.
            raise ValueError(
                f"scope filter {sorted(ALLOWED_SOURCE_TYPES)} excluded every chunk "
                f"(corpus has {len(corpus)} chunks across source types "
                f"{sorted({c.source_type for c in corpus.chunks})})"
            )
        if cfg.use_bm25:
            retrievers["sparse"] = _BM25Retriever(
                build_index(scoped.chunks, k1=cfg.bm25_k1, b=cfg.bm25_b)
            )
        if cfg.use_dense:
            emb = (embedder or Embedder(settings)).fit(scoped.chunks)
            retrievers["dense"] = _DenseRetriever(emb, scoped.chunks)
        return retrievers

    if settings.vector_store_enabled and cfg.use_dense:
        from qdrant_client import QdrantClient

        if settings.qdrant_local_path:
            client = QdrantClient(path=settings.qdrant_local_path)
        else:
            client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        emb = (embedder or Embedder(settings))
        retrievers["dense"] = _QdrantRetriever(
            client, settings.qdrant_collection, emb, ALLOWED_SOURCE_TYPES
        )

    return retrievers


@dataclass
class HybridRetriever:
    """Sparse + dense retrieval, fused, reranked, with adaptive escalation."""

    settings: Settings
    retrievers: dict[str, Retriever]
    consider_multi_query: bool = True
    status_policy: StatusPolicy | None = None

    def __post_init__(self) -> None:
        # Derived from settings when not passed explicitly, so existing call
        # sites get status awareness without changing, and an explicit policy
        # (including an inert one for ablation) still overrides it.
        if self.status_policy is None:
            self.status_policy = StatusPolicy.from_settings(self.settings)

    @property
    def strategy(self) -> str:
        if "sparse" in self.retrievers and "dense" in self.retrievers:
            return "hybrid"
        return "dense" if "dense" in self.retrievers else "bm25"

    def retrieve(
        self,
        analysis: QueryAnalysis | str,
        *,
        candidate_k: int | None = None,
        final_k: int | None = None,
        adaptive: bool | None = None,
    ) -> RetrievalResult:
        """One full retrieval pass, escalating if confidence is low.

        Accepts either a prepared `QueryAnalysis` or a plain string. A string is
        run through the lexical query analysis automatically: requiring every
        caller to import `QueryAnalysis` and call `analyse_query` first makes the
        common case (one query, no conversation history) needlessly awkward, and
        the two-step version is easy to forget, which silently skips query
        expansion rather than failing loudly.

        Returns a `RetrievalResult` in every case, including the abstaining one.
        The caller decides what to do with low confidence; this function's job is
        to report it accurately and to have already tried to fix it.
        """
        if isinstance(analysis, str):
            analysis = analyse_query(self.settings, analysis)
        cfg = self.settings.retrieval
        candidate_k = candidate_k or cfg.candidate_k
        final_k = final_k or cfg.final_k
        use_adaptive = cfg.adaptive if adaptive is None else adaptive

        started = time.perf_counter()
        per_stage: dict[str, float] = {}
        escalations = 0
        queries = list(analysis.all_queries) if self.consider_multi_query else [analysis.rewritten]

        stage = time.perf_counter()
        sparse, dense = self._score(queries, candidate_k)
        per_stage["retrieve"] = (time.perf_counter() - stage) * 1000.0

        fused = self._apply_status(self._fuse(sparse, dense, final_k), analysis.rewritten)
        considered = len(fused)

        stage = time.perf_counter()
        ranked, rerank_strategy, _ = rerank(
            self.settings, analysis.rewritten, fused, final_k
        )
        per_stage["rerank"] = (time.perf_counter() - stage) * 1000.0

        signals = estimate_confidence(
            ranked,
            sparse_ranking=_flatten(sparse),
            dense_ranking=_flatten(dense),
            query=analysis.original,
        )

        # --- Adaptive escalation ------------------------------------------
        # Widen the pool and add the remaining query variants. Bounded by
        # max_escalations so the cost is predictable.
        while use_adaptive and should_escalate(
            signals, cfg.confidence_threshold, escalations, cfg.max_escalations
        ):
            escalations += 1
            stage = time.perf_counter()
            wider = min(candidate_k * 3, max(candidate_k + 10, candidate_k * 2))
            extra_queries = queries[1:] or [analysis.original]
            sparse2, dense2 = self._score(extra_queries, wider)
            per_stage[f"escalate_{escalations}"] = (time.perf_counter() - stage) * 1000.0

            # `_score` returns one ranking *per query*, so the escalation pool is
            # the concatenation of the original per-query rankings and the new
            # ones. `_fuse` pools and fuses internally, exactly as it does on the
            # first pass, so the two passes cannot diverge. Passing
            # `[sparse, sparse2]` instead would hand `_merge_many` a list of
            # lists-of-lists and index into it as if each element were a Chunk.
            combined_sparse = [*sparse, *sparse2]
            combined_dense = [*dense, *dense2]
            candidate_pool = self._apply_status(
                self._fuse(combined_sparse, combined_dense, wider), analysis.rewritten
            )
            considered = max(considered, len(candidate_pool))

            ranked, rerank_strategy, _ = rerank(
                self.settings, analysis.rewritten, candidate_pool, final_k
            )
            signals = estimate_confidence(
                ranked,
                sparse_ranking=_flatten(combined_sparse),
                dense_ranking=_flatten(combined_dense),
                query=analysis.original,
            )

        return RetrievalResult(
            query=analysis.rewritten,
            chunks=renumber(ranked),
            strategy=self.strategy,
            rerank_strategy=rerank_strategy,
            confidence=signals.confidence,
            confidence_signals=signals.to_dict(),
            escalations=escalations,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            per_stage_ms=per_stage,
            considered=considered,
        )

    # -- internals -------------------------------------------------------- #

    def _apply_status(self, ranked: Sequence[Chunk], query: str = "") -> list[Chunk]:
        """Drop superseded documents and demote drafts.

        Applied to the fused candidate pool *before* reranking, so the reranker
        only ever sees documents that are eligible to be returned. This ordering
        is load-bearing and was previously wrong: filtering after reranking let
        superseded documents consume the whole `final_k` window, so removing them
        emptied the result instead of promoting the next candidates. On a corpus
        where the top five hits are all `status: superseded`, that returned zero
        results and the retriever looked broken rather than correctly filtered.

        Demotion is applied to `rerank_score` when present, falling back to
        `fused_score`. The choice matters: multiplying whichever field the
        *current* stage produced keeps the demotion on the same scale as the
        ranking it is adjusting, so the re-sort below is a genuine reordering
        rather than a comparison between two different scoring scales.

        `query` is passed so the policy can suspend the demotion when the
        question is about preliminary material. See `StatusPolicy`.
        """
        policy = self.status_policy
        if policy is None:
            return list(ranked)

        kept: list[Chunk] = []
        for chunk in ranked:
            if policy.is_excluded(chunk.metadata):
                continue
            multiplier = policy.multiplier(chunk.metadata, query)
            if multiplier == 1.0:
                kept.append(chunk)
                continue
            if chunk.rerank_score is not None:
                kept.append(chunk.with_scores(rerank_score=chunk.rerank_score * multiplier))
            elif chunk.fused_score is not None:
                kept.append(chunk.with_scores(fused_score=chunk.fused_score * multiplier))
            else:
                # No score to adjust. Keep the chunk at its current position
                # rather than dropping it: a demotion that silently deletes a
                # relevant draft is the failure mode `StatusPolicy` exists to
                # avoid, and an unscoreable candidate is more likely a pipeline
                # quirk than a real relevance judgement.
                kept.append(chunk)

        kept.sort(
            key=lambda c: -(
                c.rerank_score if c.rerank_score is not None else (c.fused_score or 0.0)
            )
        )
        return kept

    def _score(
        self, queries: Sequence[str], top_k: int
    ) -> tuple[list[list[Chunk]], list[list[Chunk]]]:
        """One ranked list per retriever per query."""
        sparse_lists: list[list[Chunk]] = []
        dense_lists: list[list[Chunk]] = []
        for query in queries:
            if "sparse" in self.retrievers:
                sparse_lists.append(self.retrievers["sparse"].search(query, top_k))
            if "dense" in self.retrievers:
                try:
                    dense_lists.append(self.retrievers["dense"].search(query, top_k))
                except Exception:
                    # take down a hybrid query; the sparse side still works.
                    dense_lists.append([])
        return sparse_lists, dense_lists

    def _fuse(
        self,
        sparse_lists: Sequence[Sequence[Chunk]],
        dense_lists: Sequence[Sequence[Chunk]],
        final_k: int,
    ) -> list[Chunk]:
        cfg = self.settings.retrieval
        pooled_dense = _merge_many(dense_lists, cfg.candidate_k)
        pooled_sparse = _merge_many(sparse_lists, cfg.candidate_k)

        if pooled_dense and pooled_sparse:
            return fusion.fuse(
                [pooled_dense, pooled_sparse],
                cfg.fusion,
                final_k=max(final_k, cfg.candidate_k),
                rrf_k=cfg.fusion_rrf_k,
                dense_weight=cfg.dense_weight,
                sparse_weight=cfg.sparse_weight,
            )
        if pooled_dense:
            return pooled_dense[:final_k]
        return pooled_sparse[:final_k]


def _merge_many(lists: Sequence[Sequence[Chunk]], top_k: int) -> list[Chunk]:
    """Interleave several rankings of the same corpus into one candidate list.

    Round-robin by rank rather than concatenate-and-sort: concatenating would
    let the first query's entire result list occupy the whole pool, so the second
    query could never contribute. Round-robin guarantees each query contributes
    at every depth.
    """
    if not lists:
        return []
    if len(lists) == 1:
        return list(lists[0][:top_k])

    seen: set[str] = set()
    out: list[Chunk] = []
    depth = max(len(r) for r in lists)
    for rank in range(depth):
        for ranking in lists:
            if rank < len(ranking):
                chunk = ranking[rank]
                if chunk.chunk_id not in seen:
                    seen.add(chunk.chunk_id)
                    out.append(chunk)
                    if len(out) >= top_k:
                        return out
    return out


def _flatten(lists: Sequence[Sequence[Chunk]]) -> list[Chunk]:
    out: list[Chunk] = []
    seen: set[str] = set()
    for ranking in lists:
        for chunk in ranking:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                out.append(chunk)
    return out
