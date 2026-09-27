"""HTTP surface.

Design commitments
------------------

**No agent graph.** There is one retrieval pass with an optional confidence
escalation, and that is all. A `/graph` endpoint that streams a model's chain of
thought invites users to read reasoning as an explanation of the answer, when it
is really just the model talking. The trace returned here is *pipeline* state --
which documents, which scores, which policy decisions -- because that is the part
that actually explains a result and can be checked.

**The evidence is in the response, always.** A RAG endpoint that returns text and
a confidence number is asking the caller to trust it. Returning the retrieved
spans, the policy decisions and the abstention reason makes the answer auditable
without server access, which is the difference between a system you can debug and
one you can only observe.

**Abstention is a first-class response, not an error.** It comes back 200 with
`abstained: true`. Returning 4xx would make "I don't know" indistinguishable from
"your request was malformed", and callers retry errors.

**Degradation is visible.** If the dense leg is unavailable the service says so in
`degradations` rather than silently returning lexical-only results, because a
silently degraded system is indistinguishable from a working one until quality
drops and nobody knows why.

Run it with:  uvicorn app.main:app --port 8000
"""

from __future__ import annotations

import os
import pathlib
import sys
import time
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.answering import build_answer  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.corpus.manifest import attach_chunks, build_manifest  # noqa: E402
from app.retrieval.corpus import load_corpus  # noqa: E402
from app.retrieval.pipeline import HybridRetriever, build_retrievers  # noqa: E402
from app.security.access import Principal, authorise  # noqa: E402
from app.security.injection import sanitize_for_context, scan  # noqa: E402
from app.services.security import enforce_rate_limit, verify_api_key  # noqa: E402

APP_TITLE = "Enterprise RAG"
APP_DESCRIPTION = (
    "Citation-grounded retrieval over a versioned document corpus, with "
    "reproducible quality gates and regression attribution."
)

# The single corpus-path variable. `RAG_CORPUS_ROOT` was a second, API-only
# spelling of the same setting; two names for one value means half the
# documentation is wrong for whichever path the reader is looking at.
#
# Defaults to the generated corpus, which ships seeded and reproduces
# byte-identically, so `uvicorn app.main:app` serves a real, evaluated corpus
# with no setup. `Settings.corpus_dir` is deliberately *not* consulted here: its
# default is the small `DATA/true_data` smoke corpus used by the scripts, and
# inheriting that would silently replace the evaluated corpus with six
# documents.
CORPUS_ROOT = os.environ.get("RAG_CORPUS_DIR", "evaluation/generated/docs")


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="The question.")
    tenant: str = Field("public", description="Tenant scope the caller may read.")
    subject: str = Field("public", description="Authenticated subject identifier.")
    final_k: int | None = Field(None, ge=1, le=50)
    include_evidence: bool = Field(True, description="Return the retrieved spans.")


class AskResponse(BaseModel):
    query: str
    # The scope this answer was actually produced under. Echoed back so that an
    # audit log records which tenant's view of the corpus the caller received --
    # without it, a captured response cannot be reconstructed or audited.
    subject: str
    tenants: list[str] = Field(default_factory=list)
    answer: str
    abstained: bool
    reason: str
    citations: list[dict] = Field(default_factory=list)
    confidence: float
    strategy: str
    latency_ms: float
    retrieval_ms: float
    escalations: int
    withheld_count: int
    injection_findings: int
    degradations: list[str] = Field(default_factory=list)
    evidence: list[dict] = Field(default_factory=list)
    corpus_manifest_id: str = ""


class HealthResponse(BaseModel):
    status: str
    corpus_manifest_id: str
    documents: int
    chunks: int
    retrievers: list[str]
    strategy: str
    degradations: list[str]


app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version="2.0.0")

# Built once at import: loading and indexing a corpus per request would dominate
# latency and make the reported p95 meaningless. A multi-process deployment
# therefore pays this cost once per worker, which is stated rather than hidden.
_STATE: dict = {
    "retriever": None,
    "manifest": None,
    "settings": None,
    "corpus": None,
    "degradations": [],
    "error": None,
}


def _get_state() -> dict:
    """Lazily build the index, recording the failure rather than raising.

    Building at import time would make the service unimportable when the corpus
    is missing, which turns a configuration problem into a crash loop and hides
    the actual error behind a stack trace. Deferring it lets `/health` report
    the real cause.
    """
    if _STATE["retriever"] is None and _STATE["error"] is None:
        try:
            settings = get_settings()
            # The previous code read `getattr(settings, "corpus_root", None)`,
            # but `Settings` has no `corpus_root` field -- it has `corpus_dir`.
            # The getattr therefore always returned None and every deployment
            # silently served `evaluation/generated/docs` regardless of its
            # configuration. A `getattr` on a misspelled field is a silent
            # default, and a silent default in a corpus path means serving the
            # wrong documents.
            #
            # `Settings.corpus_dir` is deliberately not consulted: its default is
            # the small `DATA/true_data` smoke corpus used by the scripts, and
            # inheriting that would silently replace the evaluated corpus with
            # six documents. `CORPUS_ROOT` already folds in RAG_CORPUS_DIR.
            root = pathlib.Path(CORPUS_ROOT)
            corpus = load_corpus(root, source_type="true")
            # Must be built from the *chunked* corpus, not the raw directory.
            # `build_manifest` alone hashes documents only, so a chunker change
            # would leave the served manifest id identical while retrieval
            # behaviour changed -- the exact case `attach_chunks` exists to make
            # visible. Building it this way also makes the id served by /health
            # identical to the one recorded in evaluation runs, so a benchmark
            # can be tied back to the exact corpus that answered a query.
            manifest = attach_chunks(build_manifest(root), corpus.chunks)
            retrievers = build_retrievers(settings, corpus)

            degradations: list[str] = []
            if corpus.chunks and "dense" not in retrievers:
                degradations.append("dense retriever unavailable; serving lexical-only results")
            if corpus.chunks and "sparse" not in retrievers:
                degradations.append("lexical retriever unavailable; serving dense-only results")

            _STATE.update(
                settings=settings,
                retriever=HybridRetriever(settings=settings, retrievers=retrievers),
                manifest=manifest,
                corpus=corpus,
                degradations=degradations,
            )
        except Exception as error:  # noqa: BLE001 - surfaced via /health, not swallowed
            _STATE["error"] = f"{type(error).__name__}: {error}"
    return _STATE


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Return the error type and message, never a stack trace.

    A stack trace in an API response is a disclosure vulnerability: it prints
    absolute paths, library versions and the shape of the internals to anyone who
    can provoke an error, which is anyone who sends a malformed request.
    """
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": f"{type(exc).__name__}: {exc}"},
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Readiness, plus the facts needed to tell a degraded system from a broken one."""
    state = _get_state()
    if state["error"]:
        return HealthResponse(
            status="degraded",
            corpus_manifest_id="",
            documents=0,
            chunks=0,
            retrievers=[],
            strategy="none",
            degradations=[f"corpus failed to load: {state['error']}"],
        )

    corpus = state["corpus"]
    return HealthResponse(
        status="ok" if not state["degradations"] else "degraded",
        corpus_manifest_id=state["manifest"].manifest_id,
        documents=len(state["manifest"]),
        chunks=len(corpus) if corpus else 0,
        retrievers=sorted(state["retriever"].retrievers),
        strategy=state["retriever"].strategy,
        degradations=list(state["degradations"]),
    )


@app.post("/ask", response_model=AskResponse, tags=["retrieval"])
def ask(
    payload: AskRequest,
    api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    _: None = Depends(verify_api_key),
    __: None = Depends(enforce_rate_limit),
) -> AskResponse:
    """Answer a question, or refuse with a reason and a count of what was withheld."""
    state = _get_state()
    if state["error"]:
        raise HTTPException(status_code=503, detail=f"corpus unavailable: {state['error']}")

    settings = state["settings"]
    started = time.perf_counter()
    result = state["retriever"].retrieve(payload.query, final_k=payload.final_k)

    # 1. Authorisation. NOTE: in this reference build the tenant comes from the
    #    request body. That is a demonstration of the *policy*, not a usable
    #    control -- a real deployment must derive the principal from verified
    #    token claims, because a client-supplied tenant string is a client-
    #    supplied permission. It is the single most common way an ACL in a RAG
    #    system becomes decorative, so it is flagged here rather than buried.
    principal = Principal(subject=payload.subject, tenants=frozenset({payload.tenant, "public"}))
    access = authorise(result.chunks, principal)
    visible = tuple(access.decision.allowed)

    # 2. Injection scan over what the caller may actually see.
    findings = scan(visible)

    # 3. Two renderings, deliberately kept separate.
    #
    #    `context` is what a provider would be *prompted* with: delimited and
    #    injection-scrubbed. `draft` is the reference build's answer: the raw
    #    retrieved text.
    #
    #    They must not be the same string. Passing the delimited block through
    #    citation verification fails every sentence, because the delimiter
    #    tokens are not in the evidence they are compared against -- the verifier
    #    correctly concludes the answer is untraceable and the system withdraws a
    #    perfectly good answer. The delimiters are a transport concern; the
    #    answer is not.
    context = sanitize_for_context(visible)
    draft = "\n\n".join(chunk.text for chunk in visible[:2])

    if access.should_refuse:
        answer = build_answer(
            payload.query,
            (),
            abstained=True,
            reason=access.reason,
            confidence=float(getattr(result, "confidence", 0.0) or 0.0),
            strategy=result.strategy,
            escalations=result.escalations,
            retrieval_ms=result.latency_ms,
            withheld_count=access.decision.withheld_count,
            injection_findings=len(findings.findings),
        )
    else:
        threshold = settings.retrieval.abstain_below
        confidence = float(getattr(result, "confidence", 0.0) or 0.0)
        low_confidence = confidence < threshold

        # Extractive by default: the verified evidence *is* the answer, and
        # `context` above is the provider prompt. A model call goes here, with
        # `context` as input; keeping the generation call swappable while
        # verification stays provider-independent is the point -- the generation
        # call is a detail, and everything that can fail silently is not.
        answer = build_answer(
            payload.query,
            visible,
            generated=draft,
            abstained=low_confidence or not visible,
            reason=(
                f"confidence {confidence:.3f} below abstain threshold {threshold:.3f}"
                if low_confidence
                else ""
            ),
            confidence=confidence,
            strategy=result.strategy,
            escalations=result.escalations,
            retrieval_ms=result.latency_ms,
            withheld_count=access.decision.withheld_count,
            injection_findings=len(findings.findings),
        )

    evidence = (
        [
            {
                "source": chunk.source,
                "chunk": int(chunk.ordinal),
                "chunk_id": chunk.chunk_id,
                "score": chunk.rerank_score if chunk.rerank_score is not None else chunk.fused_score,
                "tenant": chunk.metadata.get("tenant", "public"),
                "status": chunk.metadata.get("status", ""),
                "excerpt": chunk.text[:400],
            }
            for chunk in visible
        ]
        if payload.include_evidence
        else []
    )

    return AskResponse(
        query=payload.query,
        subject=principal.subject,
        tenants=sorted(principal.tenants),
        answer=answer.text,
        abstained=answer.abstained,
        reason=answer.reason,
        citations=[c.to_dict() for c in answer.citations],
        confidence=answer.confidence,
        strategy=answer.strategy,
        latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
        retrieval_ms=round(answer.retrieval_ms, 2),
        escalations=answer.escalations,
        withheld_count=answer.withheld_count,
        injection_findings=answer.injection_findings,
        degradations=list(state["degradations"]),
        evidence=evidence,
        corpus_manifest_id=state["manifest"].manifest_id,
    )


@app.get("/corpus", tags=["ops"])
def corpus_manifest() -> dict:
    """The corpus version currently being served.

    Exposed because a bug report that does not include the manifest id is not
    reproducible: the corpus may have changed since the report was filed.
    """
    state = _get_state()
    if state["error"]:
        raise HTTPException(status_code=503, detail=state["error"])
    return state["manifest"].to_dict()
