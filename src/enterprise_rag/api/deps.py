"""Process state: the corpus, the index, the pipeline, and how to rebuild them.

The index is built once per process and reused. Rebuilding per request would pay
corpus loading, SVD fitting and retriever construction on every call, which
would dominate latency and make the reported p95 meaningless. The cost of that
choice is that a multi-process deployment holds N indexes -- stated here rather
than left for an operator to discover.

Building is deferred to first use rather than done at import, so a missing or
malformed corpus produces a `/health` that names the cause instead of a crash
loop that hides it.
"""

from __future__ import annotations

import pathlib
import threading
import time

from enterprise_rag.config.settings import Settings, get_settings
from enterprise_rag.indexing.corpus import load_corpus
from enterprise_rag.indexing.manifest import attach_chunks, build_manifest
from enterprise_rag.observability import tracing
from enterprise_rag.pipelines.rag import RagPipeline
from enterprise_rag.pipelines.retrieval import HybridRetriever, build_retrievers

__all__ = ["ServiceState", "state"]


class ServiceState:
    """Holds the built index and rebuilds it on demand.

    A lock rather than a bare flag, because two concurrent first requests would
    otherwise both build the index. Under a threaded ASGI server that is not
    hypothetical, and the loser of the race would overwrite a good pipeline with
    an equally good but differently-numbered one.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pipeline: RagPipeline | None = None
        self._error: str | None = None
        self._built_at: float | None = None

    # -- accessors ---------------------------------------------------------

    @property
    def pipeline(self) -> RagPipeline | None:
        if self._pipeline is None and self._error is None:
            self.build()
        return self._pipeline

    @property
    def error(self) -> str | None:
        return self._error

    def require(self) -> RagPipeline:
        """The pipeline, or raise a 503-shaped error the route turns into HTTP."""
        pipeline = self.pipeline
        if pipeline is None:
            raise ServiceUnavailable(self._error or "corpus unavailable")
        return pipeline

    # -- construction ------------------------------------------------------

    def build(self, corpus_dir: str | pathlib.Path | None = None) -> RagPipeline:
        """(Re)build the index. Returns the pipeline, or records the failure."""
        with self._lock:
            root = pathlib.Path(corpus_dir) if corpus_dir else None
            try:
                settings = get_settings()
                root = root or pathlib.Path(settings.corpus_dir)
                pipeline, degradations = build_pipeline(settings, root)
                self._pipeline = pipeline
                self._degradations = degradations
                self._error = None
                self._built_at = time.time()
            except Exception as error:  # noqa: BLE001 - reported via /health
                self._error = f"{type(error).__name__}: {error}"
                tracing.error(
                    "index build failed", error=self._error, corpus_dir=str(root or "<unset>")
                )
                raise
        return self._pipeline  # type: ignore[return-value]

    def invalidate(self) -> None:
        """Drop the index so the next request rebuilds it."""
        with self._lock:
            self._pipeline = None
            self._error = None
            self._built_at = None

    @property
    def degradations(self) -> list[str]:
        return list(getattr(self, "_degradations", []))


class ServiceUnavailable(RuntimeError):
    """The index could not be built. Routes map this to 503."""


def build_pipeline(
    settings: Settings, root: pathlib.Path
) -> tuple[RagPipeline, list[str]]:
    """Load, index, and wire a pipeline for `root`. Returns (pipeline, degradations)."""
    corpus = load_corpus(root, source_type="true")
    # The manifest must be built from the *chunked* corpus, not the raw
    # directory. `build_manifest` alone hashes documents, so a chunker change
    # would leave the manifest id identical while retrieval behaviour changed.
    # `attach_chunks` folds chunk identities in, which also means the id served
    # by /health equals the one recorded in evaluation runs, so a published
    # number can be tied to the exact corpus that produced it.
    manifest = attach_chunks(build_manifest(root), corpus.chunks)
    retrievers = build_retrievers(settings, corpus)

    degradations: list[str] = []
    if corpus.chunks and "dense" not in retrievers:
        degradations.append("dense retriever unavailable; serving lexical-only results")
    if corpus.chunks and "sparse" not in retrievers:
        degradations.append("lexical retriever unavailable; serving dense-only results")
    if not corpus.chunks:
        degradations.append(
            f"corpus at {root} produced no chunks; the service will answer nothing until it does"
        )

    pipeline = RagPipeline(
        settings=settings,
        retriever=HybridRetriever(settings=settings, retrievers=retrievers),
        manifest_id=manifest.manifest_id,
        corpus_documents=len(manifest),
        corpus_chunks=len(corpus),
    )
    return pipeline, degradations


# One instance per process. Imported rather than constructed at module scope in
# each route module, because two copies of the index would double the memory and
# make the cache hit rate meaningless.
state = ServiceState()
