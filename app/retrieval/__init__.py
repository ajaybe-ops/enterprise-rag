"""Retrieval package: sparse, dense, fusion, reranking, confidence, orchestration."""

from app.retrieval.bm25 import BM25Index, bm25_scores, build_index
from app.retrieval.confidence import (
    ConfidenceSignals,
    calibration_curve,
    estimate_confidence,
    should_abstain,
    should_escalate,
)
from app.retrieval.corpus import Corpus, load_corpus
from app.retrieval.dense import Embedder, JinaDenseRetriever, LSARetriever, embeddings_mode
from app.retrieval.fusion import fuse, rrf_fuse, weighted_fuse
from app.retrieval.pipeline import (
    ALLOWED_SOURCE_TYPES,
    Corpus,
    HybridRetriever,
    build_retrievers,
)
from app.retrieval.query import analyse_query, extract_entities
from app.retrieval.rerank import lexical_overlap_scores, rerank

__all__ = [
    "ALLOWED_SOURCE_TYPES",
    "BM25Index",
    "ConfidenceSignals",
    "Corpus",
    "Embedder",
    "HybridRetriever",
    "JinaDenseRetriever",
    "LSARetriever",
    "analyse_query",
    "bm25_scores",
    "build_retrievers",
    "calibration_curve",
    "embeddings_mode",
    "estimate_confidence",
    "extract_entities",
    "fuse",
    "build_index",
    "lexical_overlap_scores",
    "load_corpus",
    "rerank",
    "rrf_fuse",
    "should_abstain",
    "should_escalate",
    "weighted_fuse",
]
