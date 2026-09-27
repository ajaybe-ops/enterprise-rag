"""The pipeline: one path from a question to an auditable answer.

This is the module the rest of the system exists to serve. There is exactly one
of them, deliberately. A second orchestration path -- a "fast" one, an
"experimental" one, a per-tenant variant -- is where reliability work goes to
die, because it is the path that does not get the fixes, the metrics, or the
regression gate.

Stage order, and why this order
-------------------------------
    analyse -> retrieve -> authorise -> build context -> generate -> verify -> answer

Two of these orders are load-bearing and easy to get wrong:

*Authorise before context.* Context construction is what puts document text into
a prompt. Authorising after that means restricted text has already been read
into memory and formatted for a model, and "we filtered the response" has to
compensate. Filtering at one place, before the text crosses the boundary, makes
the guarantee structural.

*Verify before answer.* Discussed at length in `citations.verifier`. The short
version: verifying last produces a confident wrong answer with a real citation
on it.

Every stage's inputs and outputs are recorded on the `QueryTrace`, so a bad
answer can be explained without a debugger. The trace is the product; the
answer is a field in it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from enterprise_rag.caching.result_cache import ResultCache, cache_key
from enterprise_rag.citations.models import CitationReport
from enterprise_rag.citations.verifier import verify
from enterprise_rag.config.settings import Settings
from enterprise_rag.context.builder import ContextBlock, build_context
from enterprise_rag.generation.generators import (
    ABSTAIN_TEMPLATE,
    ExtractiveGenerator,
    GenerationResult,
    Generator,
    ProviderGenerator,
)
from enterprise_rag.observability import tracing
from enterprise_rag.pipelines.retrieval import HybridRetriever
from enterprise_rag.security.access import Principal, authorise
from enterprise_rag.types import Answer, Chunk, RetrievalResult, Usage

__all__ = ["QueryTrace", "RagPipeline", "RagResponse"]


@dataclass
class QueryTrace:
    """Per-stage record of one query. The thing you read when an answer is wrong.

    Deliberately a flat list of stage records rather than a tree: there is no
    branching and no fan-out, so a tree structure would imply a flexibility the
    pipeline does not have and would be harder to diff between runs.
    """

    query_id: str
    query: str
    subject: str = ""
    tenants: tuple[str, ...] = ()
    stages: list[dict[str, Any]] = field(default_factory=list)
    total_ms: float = 0.0

    def record(self, stage: str, ms: float, **fields: Any) -> None:
        self.stages.append({"stage": stage, "ms": round(ms, 3), **fields})

    @property
    def by_stage(self) -> dict[str, float]:
        return {s["stage"]: s["ms"] for s in self.stages}

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "subject": self.subject,
            "tenants": list(self.tenants),
            "total_ms": round(self.total_ms, 3),
            "stages": self.stages,
        }


@dataclass
class RagResponse:
    """An answer plus everything needed to audit it.

    The API returns this whole object, not just `answer.text`. A RAG endpoint
    that returns text and a confidence number is asking the caller to trust it;
    returning the evidence, the stage timings and the abstention reason makes the
    answer checkable without server access.
    """

    answer: Answer
    trace: QueryTrace
    context: ContextBlock | None = None
    citations: CitationReport | None = None
    generation: GenerationResult | None = None
    access_reason: str = ""
    withheld_count: int = 0
    injection_findings: int = 0
    degradations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer.text,
            "status": self.answer.status,
            "abstained": self.answer.status == "abstained",
            "reason": self.answer.status_detail or self.access_reason,
            # The full citation objects live on the report. `Answer.citations`
            # holds only source names, because that is all a consumer of the
            # domain type needs; the API returns the richer form, which is why
            # the report and not the answer is the source here.
            "citations": (
                [c.to_dict() for c in self.citations.citations] if self.citations else []
            ),
            "sources": list(self.answer.sources),
            "confidence": round(self.answer.confidence, 4),
            "usage": self.answer.usage.to_dict(),
            "generator": self.generation.strategy if self.generation else "",
            "generation_degraded": bool(self.generation and self.generation.degraded),
            "withheld_count": self.withheld_count,
            "injection_findings": self.injection_findings,
            "degradations": list(self.degradations),
            "context": self.context.to_dict() if self.context else None,
            "citations_report": self.citations.to_dict() if self.citations else None,
            "trace": self.trace.to_dict(),
        }


class RagPipeline:
    """Owns the retrievers, the generator, the cache and the policy.

    One instance per process, built once. Constructing it per request would pay
    corpus indexing, SVD fitting and retriever construction on every call, which
    would dominate latency and make the reported p95 meaningless.
    """

    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        *,
        manifest_id: str = "",
        corpus_documents: int = 0,
        corpus_chunks: int = 0,
        generator: Generator | None = None,
        cache: ResultCache | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.manifest_id = manifest_id
        # Corpus shape is held here rather than read back off the retriever. The
        # retriever scores chunks; it does not own the corpus, and making the
        # route reach through it to count documents is how /health and the
        # evaluation harness end up reporting different numbers.
        self.corpus_documents = corpus_documents
        self.corpus_chunks = corpus_chunks
        self.generator = generator or _default_generator(settings)
        self.cache = cache or ResultCache(
            max_entries=settings.cache_max_entries, enabled=settings.cache_enabled
        )
        self.cache.bind(manifest_id)
        self._counter = 0

    # -- public API --------------------------------------------------------

    def query(
        self,
        question: str,
        principal: Principal,
        *,
        final_k: int | None = None,
    ) -> RagResponse:
        """Answer `question` as `principal`, or explain why not."""
        self._counter += 1
        query_id = f"q{self._counter:08d}"
        started = time.perf_counter()
        trace = QueryTrace(
            query_id=query_id,
            query=question,
            subject=principal.subject,
            tenants=tuple(sorted(principal.tenants)),
        )
        response = self._execute(question, principal, trace, final_k=final_k)
        trace.total_ms = (time.perf_counter() - started) * 1000.0
        response.trace = trace
        return response

    def metrics(self) -> dict:
        """Counters for ``/metrics`` and the CLI."""
        return {
            "queries": self._counter,
            "cache": self.cache.to_dict(),
            "generator": self.generator.name,
            "corpus_manifest_id": self.manifest_id,
            "corpus_documents": self.corpus_documents,
            "corpus_chunks": self.corpus_chunks,
        }

    # -- stages ------------------------------------------------------------

    def _execute(
        self,
        question: str,
        principal: Principal,
        trace: QueryTrace,
        *,
        final_k: int | None,
    ) -> RagResponse:
        degradations: list[str] = []
        usage = Usage()

        # 1. Retrieve, through the corpus-identity-keyed cache.
        stage = time.perf_counter()
        key = cache_key(
            question, principal.tenants, self.manifest_id, final_k or self.settings.retrieval.final_k
        )
        result = self.retriever.retrieve(question, final_k=final_k)
        cached = self.cache.get(key)
        if cached is not None:
            # A hit is only possible when the manifest matches, because the key
            # contains it and `bind` cleared the cache on any change. Replacing
            # the retrieved set with the cached one is therefore safe, and the
            # trace says so rather than hiding the difference.
            result = RetrievalResult(
                query=result.query,
                chunks=cached,
                strategy=result.strategy,
                rerank_strategy=result.rerank_strategy,
                confidence=result.confidence,
                confidence_signals=result.confidence_signals,
                escalations=result.escalations,
                considered=result.considered,
            )
            cache_hit = True
        else:
            self.cache.put(key, result.chunks)
            cache_hit = False
        trace.record(
            "retrieve",
            (time.perf_counter() - stage) * 1000.0,
            strategy=result.strategy,
            rerank=result.rerank_strategy,
            returned=len(result.chunks),
            considered=result.considered,
            confidence=round(result.confidence, 4),
            escalations=result.escalations,
            cache_hit=cache_hit,
            per_stage_ms={k: round(v, 2) for k, v in result.per_stage_ms.items()},
        )

        # 2. Authorise, before any retrieved text reaches a prompt.
        stage = time.perf_counter()
        access = authorise(result.chunks, principal)
        visible: tuple[Chunk, ...] = tuple(access.decision.allowed)
        trace.record(
            "authorise",
            (time.perf_counter() - stage) * 1000.0,
            withheld=access.decision.withheld_count,
            withheld_sources=list(access.decision.withheld_sources),
            should_refuse=access.should_refuse,
            withheld_at_rank=access.evidence_withheld_at_rank,
        )

        if access.should_refuse or not visible:
            return self._abstain(
                question=question,
                reason=access.reason or "no readable evidence within the caller's scope",
                result=result,
                usage=usage,
                trace=trace,
                access_reason=access.reason,
                withheld_count=access.decision.withheld_count,
                degradations=degradations,
            )

        # 3. Build the context: dedup, budget, fence, redact.
        stage = time.perf_counter()
        context = build_context(
            visible,
            max_chars=self.settings.max_context_chars,
            max_chunks=min(self.settings.max_chunks, len(visible)),
        )
        trace.record(
            "context",
            (time.perf_counter() - stage) * 1000.0,
            included=len(context.chunks),
            dropped_budget=context.dropped_budget,
            dropped_duplicate=context.dropped_duplicate,
            chars=context.chars,
            utilization=round(context.utilization, 4),
            injection_findings=len(context.injection.findings) if context.injection else 0,
        )
        findings = len(context.injection.findings) if context.injection else 0

        # 4. Generate. A provider failure degrades to extraction rather than
        #    failing the request: an unconfigured dependency is a normal state.
        stage = time.perf_counter()
        generation = self.generator.generate(question, context)
        if generation.degraded:
            degradations.append(generation.reason)
            generation = ExtractiveGenerator().generate(question, context)
            generation = GenerationResult(
                text=generation.text,
                usage=generation.usage,
                strategy=generation.strategy,
                degraded=True,
                reason=generation.strategy + " fallback after provider failure",
            )
        usage = usage + generation.usage
        trace.record(
            "generate",
            (time.perf_counter() - stage) * 1000.0,
            strategy=generation.strategy,
            degraded=generation.degraded,
            output_chars=len(generation.text),
            prompt_tokens=generation.usage.prompt_tokens,
            completion_tokens=generation.usage.completion_tokens,
            cost_usd=round(generation.usage.cost_usd, 6),
            tokens_estimated=generation.usage.estimated,
        )

        if not generation.text.strip():
            return self._abstain(
                question=question,
                reason=generation.reason or "the generator produced no text for the retrieved evidence",
                result=result,
                usage=usage,
                trace=trace,
                context=context,
                generation=generation,
                access_reason=access.reason,
                withheld_count=access.decision.withheld_count,
                injection_findings=findings,
                degradations=degradations,
            )

        # 5. Verify, before the answer is presented.
        stage = time.perf_counter()
        report = verify(generation.text, context.chunks)
        trace.record(
            "verify",
            (time.perf_counter() - stage) * 1000.0,
            sentences=report.total_sentences,
            verified=len(report.verified_sentences),
            rejected=len(report.rejected_sentences),
            coverage=round(report.coverage, 4),
        )

        if not report.supported:
            # Withdrawing rather than serving the supported subset: a partial
            # answer presented as complete is the failure most likely to mislead
            # without being noticed.
            return self._abstain(
                question=question,
                reason=(
                    f"none of the {report.total_sentences} generated sentence(s) could be "
                    f"traced to the retrieved evidence; withdrawn rather than serving an "
                    f"unattributable answer"
                ),
                result=result,
                usage=usage,
                trace=trace,
                context=context,
                generation=generation,
                citations=report,
                access_reason=access.reason,
                withheld_count=access.decision.withheld_count,
                injection_findings=findings,
                degradations=degradations,
            )

        # 6. Answer.
        answer = Answer(
            text=" ".join(report.verified_sentences)
            + (
                f"\n\n({len(report.rejected_sentences)} unsupported sentence(s) removed.)"
                if report.rejected_sentences
                else ""
            ),
            status="answered",
            citations=tuple(c.source for c in report.citations),
            citation_report=report.to_dict(),
            usage=usage,
            sources=tuple(context.sources),
            confidence=result.confidence,
            retrieval=result,
        )
        return RagResponse(
            answer=answer,
            trace=trace,
            context=context,
            citations=report,
            generation=generation,
            access_reason=access.reason,
            withheld_count=access.decision.withheld_count,
            injection_findings=findings,
            degradations=degradations,
        )

    def _abstain(
        self,
        *,
        question: str,
        reason: str,
        result: RetrievalResult,
        usage: Usage,
        trace: QueryTrace,
        context: ContextBlock | None = None,
        generation: GenerationResult | None = None,
        citations: CitationReport | None = None,
        access_reason: str = "",
        withheld_count: int = 0,
        injection_findings: int = 0,
        degradations: Sequence[str] = (),
    ) -> RagResponse:
        """Build the refusal. Abstention is a 200 with a reason, not an error."""
        trace.record("abstain", 0.0, reason=reason)
        tracing.info(
            "abstained",
            query_id=trace.query_id,
            subject=trace.subject,
            reason=reason,
            confidence=round(result.confidence, 4),
        )
        return RagResponse(
            answer=Answer(
                text=ABSTAIN_TEMPLATE,
                status="abstained",
                status_detail=reason,
                confidence=result.confidence,
                usage=usage,
                retrieval=result,
            ),
            trace=trace,
            context=context,
            citations=citations,
            generation=generation,
            access_reason=access_reason,
            withheld_count=withheld_count,
            injection_findings=injection_findings,
            degradations=list(degradations),
        )


def _default_generator(settings: Settings) -> Generator:
    """Provider when configured, deterministic extraction otherwise.

    The extractive path is what produced the committed evaluation results, so
    "no credentials" is a supported mode rather than a degraded one.
    """
    if settings.llm_enabled:
        return ProviderGenerator(settings)
    return ExtractiveGenerator()
