"""Domain types shared across retrieval, generation and evaluation.

`Chunk` is the reason the previous reranking was broken. The old retriever
threaded bare `str` through the pipeline and re-attached provenance by matching
chunk *text* against a set:

    kept = {chunk for chunk in ranked}          # list[str] -> set[str]
    formatted = [f"SOURCE: {d['source']}..." for d in raw_results
                 if d["content"] in kept]       # keeps Qdrant's order

Two things went wrong. The output order came from `raw_results`, so the
cross-encoder's ranking was thrown away — it only decided *which* documents
survived, never their order. And matching on text alone means two chunks with
identical content both match, so provenance can attach to the wrong source.

A single immutable `Chunk` carrying its own identity, provenance and every
score makes both errors unrepresentable: order comes from the list the ranker
returns, and the source always travels with the content it belongs to.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

__all__ = ["Answer", "Chunk", "QueryAnalysis", "RetrievalResult", "Usage"]

RetrievalStrategy = Literal["bm25", "dense", "hybrid"]
RerankStrategy = Literal["none", "jina", "lexical_overlap"]


def stable_chunk_id(source: str, ordinal: int, text: str) -> str:
    """Deterministic id for a chunk.

    Deriving the id from (source, ordinal, content-hash) rather than a random
    UUID means re-ingesting a document upserts onto the same points instead of
    appending duplicates — which is what `--force` used to do.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return hashlib.sha256(f"{source}:{ordinal}:{digest}".encode()).hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable unit of text, with its provenance and scores.

    `rank` is the single source of truth for ordering. Retrievers return chunks
    with `rank` set; fusion re-ranks; the reranker re-ranks. Nobody re-derives
    order from a parallel structure.
    """

    chunk_id: str
    text: str
    source: str
    source_type: str = "true"
    ordinal: int = 0
    char_start: int = 0
    metadata: dict[str, str] = field(default_factory=dict)

    # Populated as the chunk moves through the pipeline. `None` means "this
    # stage did not run", which is different from "this stage scored 0".
    bm25_score: float | None = None
    dense_score: float | None = None
    rerank_score: float | None = None
    # Fusion output. Kept alongside the per-retriever scores rather than
    # replacing them, so a benchmark can still attribute *why* a document won.
    fused_score: float | None = None
    rank: int | None = None
    vector_id: int | None = None

    @classmethod
    def from_text(
        cls,
        text: str,
        source: str,
        ordinal: int = 0,
        source_type: str = "true",
        char_start: int = 0,
        **metadata: str,
    ) -> Chunk:
        return cls(
            chunk_id=stable_chunk_id(source, ordinal, text),
            text=text,
            source=source,
            source_type=source_type,
            ordinal=ordinal,
            char_start=char_start,
            metadata=dict(metadata),
        )

    @property
    def citation_label(self) -> str:
        return self.source

    def with_scores(self, **kwargs) -> Chunk:
        return replace(self, **kwargs)

    def to_payload(self) -> dict:
        """Qdrant payload. `char_start` is a real offset, not a chunk index."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "source_type": self.source_type,
            "ordinal": self.ordinal,
            "char_start": self.char_start,
            "metadata": self.metadata,
        }

    @classmethod
    def from_payload(cls, payload: dict, vector_id: int | None = None, score: float = 0.0):
        return cls(
            chunk_id=payload.get("chunk_id") or stable_chunk_id(
                payload.get("source", "unknown"),
                int(payload.get("ordinal", 0)),
                payload.get("text", ""),
            ),
            text=payload.get("text", ""),
            source=payload.get("source", "unknown"),
            source_type=payload.get("source_type", "true"),
            ordinal=int(payload.get("ordinal", 0)),
            char_start=int(payload.get("char_start", 0)),
            metadata=payload.get("metadata") or {},
            dense_score=score,
            vector_id=vector_id,
        )

    def preview(self, width: int = 80) -> str:
        flat = " ".join(self.text.split())
        return flat if len(flat) <= width else flat[: width - 1] + "…"


@dataclass(frozen=True, slots=True)
class QueryAnalysis:
    """Result of query transformation. Every field has a safe default.

    The pipeline must work when no LLM is configured, so `rewritten` falls back
    to the original question and `variants` to a single lexical variant.
    """

    original: str
    rewritten: str
    variants: tuple[str, ...] = ()
    is_conversational: bool = False
    entities: tuple[str, ...] = ()
    method: str = "identity"

    @property
    def all_queries(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for q in (self.rewritten, *self.variants):
            if q and q.strip():
                seen.setdefault(q.strip(), None)
        return tuple(seen)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Everything one retrieval pass produced, including the evidence needed to
    decide whether to trust it."""

    query: str
    chunks: tuple[Chunk, ...]
    strategy: RetrievalStrategy = "hybrid"
    rerank_strategy: RerankStrategy = "none"
    confidence: float = 0.0
    confidence_signals: dict[str, float] = field(default_factory=dict)
    escalations: int = 0
    latency_ms: float = 0.0
    per_stage_ms: dict[str, float] = field(default_factory=dict)
    considered: int = 0

    def __len__(self) -> int:
        return len(self.chunks)

    def __iter__(self):
        return iter(self.chunks)

    @property
    def sources(self) -> list[str]:
        out: list[str] = []
        for chunk in self.chunks:
            if chunk.source not in out:
                out.append(chunk.source)
        return out

    @property
    def top_score(self) -> float | None:
        for chunk in self.chunks:
            if chunk.rerank_score is not None:
                return chunk.rerank_score
        if self.chunks and self.chunks[0].fused_score is not None:
            return self.chunks[0].fused_score
        for chunk in self.chunks:
            if chunk.dense_score is not None:
                return chunk.dense_score
        if self.chunks and self.chunks[0].bm25_score is not None:
            return self.chunks[0].bm25_score
        return None


@dataclass(frozen=True, slots=True)
class Usage:
    """Token and latency accounting for one LLM call.

    Counts are the provider's own numbers when available, never an estimate.
    `estimated` is True only when the provider reported no usage block, in which
    case a tokenizer approximation is used and the caller can tell the
    difference. `cost_usd` is an *estimate* derived from configurable list
    prices — see docs/08_COST_MODEL.md.
    """

    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    estimated: bool = False
    cached: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            model=self.model or other.model,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            latency_ms=self.latency_ms + other.latency_ms,
            cost_usd=self.cost_usd + other.cost_usd,
            estimated=self.estimated or other.estimated,
            cached=self.cached or other.cached,
        )

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": round(self.latency_ms, 2),
            "cost_usd": round(self.cost_usd, 6),
            "estimated": self.estimated,
            "cached": self.cached,
        }


@dataclass(frozen=True, slots=True)
class Answer:
    """Final response, with the evidence needed to audit it.

    `status` is one of "answered" | "abstained" | "refused". It is a field rather
    than an exception because a refusal is a legitimate answer to a question, not
    a failure of the request: collapsing the two makes "I don't know" look like
    a bug, and callers retry bugs.

    `status_detail` is the human-readable *why*, and it is required whenever the
    status is not "answered". An abstention with no reason is unactionable -- the
    caller cannot tell whether to rephrase, wait for an index, or give up.
    """

    text: str
    status: str = "answered"
    status_detail: str = ""
    citations: tuple[str, ...] = ()
    citation_report: dict = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    sources: tuple[str, ...] = ()
    confidence: float = 0.0
    retrieval: RetrievalResult | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "status": self.status,
            "status_detail": self.status_detail,
            "citations": list(self.citations),
            "sources": list(self.sources),
            "confidence": round(self.confidence, 4),
            "usage": self.usage.to_dict(),
            "citations_report": self.citation_report,
        }


def renumber(chunks: Iterable[Chunk]) -> tuple[Chunk, ...]:
    """Re-apply 0-based rank over `chunks` in their current order.

    0-based, consistently with the `enumerate` in the BM25, dense and fusion
    scorers. It is an internal ordering index, **not** a display position: a UI
    or citation label must use `rank + 1`, because "rank 0" is meaningless to a
    reader. Keeping the two apart is why this function exists at all rather than
    callers setting `rank` ad hoc.
    """
    return tuple(replace(chunk, rank=i) for i, chunk in enumerate(chunks))


def filter_by_source_type(
    chunks: Sequence[Chunk], allowed: frozenset[str] | None
) -> list[Chunk]:
    """Scope gate. `allowed=None` disables filtering (used by benchmarks that
    deliberately measure cross-corpus confusion)."""
    if allowed is None:
        return list(chunks)
    return [c for c in chunks if c.source_type in allowed]
