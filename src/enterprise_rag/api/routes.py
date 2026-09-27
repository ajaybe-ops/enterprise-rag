"""Routes. One module per resource, registered by `api.app`.

Routes are thin. They translate HTTP into a `RagPipeline.query` call and back,
and every decision -- authorization, abstention, degradation -- is made in the
pipeline. A route that contains a policy branch is a policy branch that the
evaluation harness cannot exercise, because the harness calls the pipeline
directly.
"""

from __future__ import annotations

import pathlib
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse

from enterprise_rag.api.deps import ServiceUnavailable, build_pipeline, state
from enterprise_rag.api.models import (
    HealthResponse,
    IndexRequest,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from enterprise_rag.api.security import enforce_rate_limit, rate_limit_state, verify_api_key
from enterprise_rag.config.settings import get_settings
from enterprise_rag.ingestion.pipeline import ingest_directory
from enterprise_rag.security.access import Principal

__all__ = ["corpus_router", "ingest_router", "metrics_router", "query_router"]

query_router = APIRouter(tags=["retrieval"])
ingest_router = APIRouter(tags=["ingestion"])
metrics_router = APIRouter(tags=["ops"])
corpus_router = APIRouter(tags=["ops"])

_AUTH = [Depends(verify_api_key), Depends(enforce_rate_limit)]


@query_router.post("/query", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    _: None = Depends(verify_api_key),
    __: None = Depends(enforce_rate_limit),
) -> QueryResponse:
    """Answer a question, or refuse with a reason and a count of what was withheld."""
    try:
        pipeline = state.require()
    except ServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"corpus unavailable: {exc}") from exc

    principal = Principal(
        subject=payload.subject, tenants=frozenset({payload.tenant, "public"})
    )
    response = pipeline.query(payload.query, principal, final_k=payload.final_k)
    payload_dict = response.to_dict()

    evidence: list[dict[str, Any]] = []
    if payload.include_evidence and response.context is not None:
        evidence = [
            {
                "source": chunk.source,
                "chunk": int(chunk.ordinal),
                "chunk_id": chunk.chunk_id,
                "rank": chunk.rank,
                "score": chunk.rerank_score if chunk.rerank_score is not None else chunk.fused_score,
                "tenant": chunk.metadata.get("tenant", "public"),
                "status": chunk.metadata.get("status", ""),
                "excerpt": chunk.text[:400],
            }
            for chunk in response.context.chunks
        ]

    return QueryResponse(
        query_id=response.trace.query_id,
        query=payload.query,
        subject=principal.subject,
        tenants=sorted(principal.tenants),
        evidence=evidence,
        corpus_manifest_id=pipeline.manifest_id,
        **{k: v for k, v in payload_dict.items() if k not in {"query", "subject", "tenants"}},
    )


@query_router.post("/ask", response_model=QueryResponse, include_in_schema=False)
def ask(payload: QueryRequest) -> QueryResponse:
    """Backwards-compatible alias for `POST /query`.

    Kept because existing clients call this path, and silently breaking them is
    worse than carrying one shim. It is `include_in_schema=False` so the
    documented surface stays a single canonical endpoint. Deprecated: use
    `POST /query`.
    """
    return query(payload)


@ingest_router.post("/ingest", response_model=IngestResponse, dependencies=_AUTH)
def ingest(payload: IngestRequest) -> IngestResponse:
    """Parse a directory into a versioned corpus, then invalidate the live index.

    Writes Markdown plus front matter into a target directory and reports the
    resulting manifest. The filesystem is treated as a cache of parsed text
    rather than as the corpus: `indexing.manifest` is the source of truth and
    can be rebuilt at any time, so an interrupted run leaves a partially
    populated directory rather than a corrupt index.

    Refuses a missing source directory with a 400 naming it, rather than
    producing an empty corpus and a manifest id that looks valid.
    """
    source = pathlib.Path(payload.source_dir)
    if not source.is_dir():
        raise HTTPException(
            status_code=400, detail=f"source_dir does not exist or is not a directory: {source}"
        )

    out_dir = pathlib.Path(payload.output_dir or f"evals/datasets/{payload.corpus_name}")
    report = ingest_directory(
        source,
        out_dir,
        tenant=payload.tenant,
        status=payload.status,
    )
    # The live index is now stale by definition. Dropping it means the next
    # query rebuilds against the new corpus, and the manifest-keyed cache
    # invalidates itself at that point.
    state.invalidate()

    written = report.get("written", [])
    skipped = report.get("skipped", [])
    warnings = [f"output directory {out_dir} already contained files not from this run"]
    if skipped:
        warnings.append(
            f"{len(skipped)} file(s) produced no text and were skipped: "
            + ", ".join(sorted(skipped)[:5])
            + ("..." if len(skipped) > 5 else "")
        )

    return IngestResponse(
        corpus_name=payload.corpus_name,
        output_dir=str(out_dir),
        documents=int(report.get("documents", 0)),
        manifest_id=str(report.get("manifest_id", "")),
        tenant=payload.tenant,
        status=payload.status,
        written=sorted(written),
        skipped=sorted(skipped),
        warnings=warnings,
    )


@ingest_router.post("/index", response_model=HealthResponse, dependencies=_AUTH)
def rebuild_index(payload: IndexRequest) -> HealthResponse:
    """Rebuild the live index, optionally against a different corpus.

    Synchronous and in-process. A long rebuild belongs in a job queue with a
    status endpoint; doing it here and returning 200 immediately would report
    success for an index that does not exist yet.
    """
    try:
        build_pipeline(get_settings(), pathlib.Path(payload.corpus_dir or get_settings().corpus_dir))
    except Exception as exc:  # noqa: BLE001 - the reason is the useful part
        raise HTTPException(status_code=503, detail=f"index rebuild failed: {exc}") from exc
    state.invalidate()
    return health()


@corpus_router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Readiness, plus the facts needed to tell a degraded system from a broken one."""
    settings = get_settings()
    pipeline = state.pipeline

    if pipeline is None:
        return HealthResponse(
            status="degraded",
            corpus_manifest_id="",
            documents=0,
            chunks=0,
            retrievers=[],
            strategy="none",
            generator="",
            capabilities=settings.capabilities(),
            config_warnings=list(settings.warnings),
            degradations=[f"corpus failed to load: {state.error}"],
        )

    retrievers = getattr(pipeline.retriever, "retrievers", {})
    return HealthResponse(
        status="ok" if not state.degradations else "degraded",
        corpus_manifest_id=pipeline.manifest_id,
        documents=pipeline.corpus_documents,
        chunks=pipeline.corpus_chunks,
        retrievers=sorted(retrievers),
        strategy=getattr(pipeline.retriever, "strategy", "unknown"),
        generator=pipeline.generator.name,
        capabilities=settings.capabilities(),
        config_warnings=list(settings.warnings),
        degradations=list(state.degradations),
    )


@corpus_router.get("/corpus")
def corpus_manifest() -> dict:
    """The corpus version currently being served.

    Exposed because a bug report without the manifest id is not reproducible: the
    corpus may have changed since it was filed.
    """
    pipeline = state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail=state.error or "corpus unavailable")
    return {
        "manifest_id": pipeline.manifest_id,
        "documents": pipeline.corpus_documents,
        "chunks": pipeline.corpus_chunks,
        "cache": pipeline.cache.to_dict(),
    }


def _prometheus(pipeline: Any) -> str:
    """Render counters in Prometheus text exposition format.

    Hand-rolled rather than pulled from a client library. The format is a dozen
    lines of string assembly, and adding a dependency for it would mean either
    trusting a transitive resolver for a scraper surface or vendoring a
    maintained version of this same loop. The HELP/TYPE pairing below is what
    makes the output valid, and it is why every metric declares its type.
    """
    lines: list[str] = [
        "# HELP rag_queries_total Queries served by this process since start.",
        "# TYPE rag_queries_total counter",
        f"rag_queries_total {pipeline.metrics()['queries']}",
        "# HELP rag_cache_hits_total Retrieval cache hits.",
        "# TYPE rag_cache_hits_total counter",
        f"rag_cache_hits_total {pipeline.cache.stats.hits}",
        "# HELP rag_cache_misses_total Retrieval cache misses.",
        "# TYPE rag_cache_misses_total counter",
        f"rag_cache_misses_total {pipeline.cache.stats.misses}",
        "# HELP rag_cache_entries Current cache occupancy.",
        "# TYPE rag_cache_entries gauge",
        f"rag_cache_entries {pipeline.cache.stats.entries}",
        "# HELP rag_cache_invalidations_total Entries dropped because the corpus changed.",
        "# TYPE rag_cache_invalidations_total counter",
        f"rag_cache_invalidations_total {pipeline.cache.stats.invalidations}",
        "# HELP rag_rate_limit_clients Clients with an open rate-limit window.",
        "# TYPE rag_rate_limit_clients gauge",
        f"rag_rate_limit_clients {len(rate_limit_state())}",
        "# HELP rag_corpus_info Corpus identity currently served; always 1.",
        "# TYPE rag_corpus_info gauge",
        'rag_corpus_info{manifest_id="%s",generator="%s"} 1'
        % (pipeline.manifest_id, pipeline.generator.name),
    ]
    return "\n".join(lines) + "\n"


@metrics_router.get("/metrics", response_class=PlainTextResponse)
def metrics() -> PlainTextResponse:
    """Prometheus scrape endpoint.

    Served as `text/plain` rather than wrapped in a JSON response model: the
    exposition format is a wire format, and a JSON object here would be a
    subtly different metrics schema that no scraper can read.

    No authentication. Standard practice is to bind the metrics port to a private
    interface or give it a separate listener, and that is the deployment's call
    to make -- see `docs/deployment.md`. What matters here is the content: the
    endpoint exposes counters and corpus identity, never query text, never
    answers, never the configuration dump.
    """
    pipeline = state.pipeline
    if pipeline is None:
        return PlainTextResponse(
            "# index unavailable\n",
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )
    return PlainTextResponse(
        _prometheus(pipeline),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
