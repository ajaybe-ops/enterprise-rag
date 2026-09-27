"""The FastAPI application factory.

Design commitments
------------------

**The evidence is in the response, always.** A RAG endpoint that returns text
and a confidence number is asking the caller to trust it. Returning the
retrieved spans, the stage timings, the authorization decision and the token
accounting makes the answer auditable without server access, which is the
difference between a system you can debug and one you can only observe.

**Abstention is a first-class response, not an error.** It comes back 200 with
`status: "abstained"` and a reason. A 4xx would make "I don't know"
indistinguishable from "your request was malformed", and callers retry errors.

**Degradation is visible.** If the dense leg is unavailable, or a provider fails
and extraction takes over, the response says so in `degradations`. A silently
degraded system is indistinguishable from a working one until quality drops and
nobody knows why.

**No stack traces in responses.** An error body is a disclosure surface: it
prints absolute paths, library versions and internal structure to anyone who can
provoke an error, which is anyone who sends a malformed request.

Run it with:  uvicorn enterprise_rag.api.app:app --port 8000
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

# Support running from a source checkout without installation, which is what
# `make dev` and the tests do. `pyproject.toml` declares the same package, so an
# installed environment takes the normal path and this is a no-op there.
_SRC = Path(__file__).resolve().parents[2]
if _SRC.name == "src" and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from enterprise_rag import __version__  # noqa: E402
from enterprise_rag.api.deps import ServiceUnavailable, state  # noqa: E402
from enterprise_rag.api.routes import (  # noqa: E402
    corpus_router,
    ingest_router,
    metrics_router,
    query_router,
)

__all__ = ["app", "create_app"]

DESCRIPTION = (
    "Citation-grounded retrieval over a versioned, content-addressed document "
    "corpus, with reproducible quality gates and regression attribution. "
    "Runs with no credentials: retrieval falls back to BM25 plus an offline LSA "
    "projection and generation to extraction."
)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
        """Build the index once, at startup, and log rather than crash on failure.

        Failing to start would turn a missing corpus into a crash loop whose real
        cause is four lines above in the log. Starting anyway lets `/health`
        report the failure precisely, so an operator can ask the service what is
        wrong with it.
        """
        try:
            state.build()
        except Exception as exc:
            print(f"[enterprise-rag] index not built: {exc}")
        yield

    application = FastAPI(
        title="Enterprise RAG",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
    )

    @application.exception_handler(ServiceUnavailable)
    async def service_unavailable(
        request: Request,  # noqa: ARG001 -- Starlette handler signature
        exc: ServiceUnavailable,
    ) -> JSONResponse:
        """The index is not built. 503, because a retry is reasonable."""
        return JSONResponse(
            status_code=503,
            content={"error": "corpus_unavailable", "detail": str(exc)},
            headers={"Retry-After": "5"},
        )

    @application.exception_handler(Exception)
    async def unhandled(
        request: Request,  # noqa: ARG001 -- Starlette handler signature
        exc: Exception,
    ) -> JSONResponse:
        """Error type and message, never a stack trace."""
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "detail": f"{type(exc).__name__}: {exc}"},
        )

    application.include_router(query_router)
    application.include_router(ingest_router)
    application.include_router(metrics_router)
    application.include_router(corpus_router)

    return application


app = create_app()
