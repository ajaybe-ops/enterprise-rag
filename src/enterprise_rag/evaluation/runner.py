"""Run a probe set and produce a reproducible report.

The report is the durable artifact, not the console output. A console line
scrolls away; a saved report can be diffed against last week's, which is the
entire point. So `RunReport` carries enough context to answer, months later and
without the original code:

    what corpus     -> manifest_id
    what questions  -> dataset_id
    what settings   -> config_fingerprint
    what it scored  -> outcomes, per obligation
    on what machine -> environment

Any one of those missing turns a regression report into guesswork. A recall
drop with the same manifest_id is a code change; with a different manifest_id it
is a corpus change; with the same dataset_id but a different fingerprint it is a
configuration change. Those have completely different fixes, and the report is
what distinguishes them.

Executors
---------
`run_probe_set` takes an executor rather than importing the retriever directly.
The measurement harness must not be entangled with the system it measures, or a
bug in the harness and a bug in the pipeline become indistinguishable. The
bundled `RetrievalExecutor` measures the retrieval stack offline with no
network and no API key, which is what makes it usable as a CI gate; a second
executor can drive the full serving path to check the two agree.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol, Sequence

from enterprise_rag.evaluation.metrics import AggregateReport, aggregate, evaluate_probe
from enterprise_rag.evaluation.probe import Probe, ProbeOutcome, ProbeSet
from enterprise_rag.security.access import Principal, authorise

__all__ = [
    "ProbeExecution",
    "ProbeExecutor",
    "RetrievalExecutor",
    "RunReport",
    "run_probe_set",
    "config_fingerprint",
]


@dataclass(frozen=True, slots=True)
class ProbeExecution:
    """What one probe actually produced, before it is scored."""

    ranked_sources: tuple[str, ...] = ()
    context_text: str = ""
    abstained: bool = False
    latency_ms: float = 0.0
    candidates_considered: int = 0
    confidence: float = 0.0
    escalations: int = 0
    strategy: str = ""
    error: str = ""


class ProbeExecutor(Protocol):
    """Executes one probe against a system under test."""

    def __call__(self, probe: Probe) -> ProbeExecution: ...


def config_fingerprint(settings: object) -> str:
    """Short hash of the retrieval-relevant configuration.

    Built from the settings' public data rather than `repr`, so it is stable
    across refactors that do not change behaviour. It is hashed rather than
    stored in full because a report should record *that* two runs differed, not
    become a second copy of the config that can itself drift out of sync.
    """
    material: dict = {}
    for name in ("retrieval", "generation", "security", "ingestion"):
        section = getattr(settings, name, None)
        if section is None:
            continue
        if isinstance(section, Mapping):
            material[name] = {k: v for k, v in sorted(section.items()) if isinstance(v, (int, float, str, bool))}
        else:
            fields = getattr(section, "__dataclass_fields__", None) or getattr(type(section), "__annotations__", {})
            material[name] = {
                k: getattr(section, k)
                for k in sorted(fields)
                if isinstance(getattr(section, k, None), (int, float, str, bool))
            }
    return hashlib.sha256(json.dumps(material, sort_keys=True, default=str).encode()).hexdigest()[:12]


@dataclass
class RetrievalExecutor:
    """Measure the in-process retrieval stack.

    This is the offline gate: no HTTP, no API key, no network. `assemble_context`
    is injected rather than imported so the metric "did the right span reach the
    context" is measured against the *same* assembly function serving uses. If
    the two ever diverge, fact_coverage silently measures a fiction -- so the
    call site passes the real one.
    """

    retriever: object
    settings: object
    top_k: int = 5
    assemble_context: Callable[[Sequence], str] | None = None
    should_abstain: Callable[[float], bool] | None = None
    tenant_filter: Callable[[Probe], Sequence[str]] | None = None
    authorise: bool = True

    def __call__(self, probe: Probe) -> ProbeExecution:
        try:
            result = self.retriever.retrieve(probe.query)
        except Exception as error:  # noqa: BLE001 - an executor must never crash a suite
            return ProbeExecution(error=f"{type(error).__name__}: {error}")

        chunks = tuple(getattr(result, "chunks", ()) or ())
        ranked = tuple(
            dict.fromkeys(chunk.source for chunk in chunks)
        )

        # Authorisation is applied to the *executed* result, not simulated
        # afterwards. Two things depend on this. A probe carrying a subject from
        # another tenant can only observe what that subject may see, so an ACL
        # bug shows up as a probe failure instead of a passing assertion. And the
        # refusal decision below is made from the real ranked list, which is the
        # only place the information exists.
        refusal_reason = ""
        if self.authorise:
            principal = Principal(
                subject=probe.subject or "anonymous",
                tenants=frozenset({probe.tenant} | {"public"} if probe.tenant != "public" else frozenset({"public"})),
            )
            outcome = authorise(chunks, principal)
            chunks = tuple(outcome.decision.allowed)
            ranked = tuple(dict.fromkeys(c.source for c in chunks))
            if outcome.should_refuse:
                refusal_reason = outcome.reason
        elif self.tenant_filter:
            # Legacy path, retained so the tenant filter can be ablated against
            # the authorisation policy. The two are not equivalent: a filter
            # cannot distinguish "restricted evidence was the answer" from
            # "restricted noise was below the fold", which is exactly the
            # difference the forbidden probes exist to measure.
            allowed = self.tenant_filter(probe)
            if allowed is not None:
                permitted = set(allowed)
                chunks = tuple(c for c in chunks if c.metadata.get("tenant", "public") in permitted)
                ranked = tuple(dict.fromkeys(c.source for c in chunks))

        context = self.assemble_context(chunks) if self.assemble_context else "\n\n".join(c.text for c in chunks)
        confidence = float(getattr(result, "confidence", 0.0) or 0.0)
        threshold = float(getattr(getattr(self.settings, "retrieval", None), "confidence_threshold", 0.0) or 0.0)
        if refusal_reason:
            # A policy refusal outranks a confidence signal. Reporting high
            # confidence here would mean the system knows it could not answer and
            # is answering anyway, which is the exact behaviour this path exists
            # to prevent.
            abstain = True
        elif self.should_abstain:
            abstain = self.should_abstain(confidence)
        else:
            abstain = confidence < threshold

        return ProbeExecution(
            ranked_sources=ranked,
            context_text=context,
            abstained=abstain,
            latency_ms=float(getattr(result, "latency_ms", 0.0) or 0.0),
            candidates_considered=int(getattr(result, "considered", 0) or 0),
            confidence=confidence,
            escalations=int(getattr(result, "escalations", 0) or 0),
            strategy=str(getattr(result, "strategy", "") or ""),
        )


@dataclass(frozen=True, slots=True)
class RunReport:
    """The durable record of one evaluation run."""

    run_id: str
    label: str
    dataset_id: str
    manifest_id: str
    config_fingerprint: str
    aggregate: AggregateReport
    outcomes: tuple[ProbeOutcome, ...]
    k: int = 5
    environment: Mapping[str, str] = field(default_factory=dict)
    notes: str = ""

    @property
    def failures(self) -> tuple[ProbeOutcome, ...]:
        return tuple(o for o in self.outcomes if not o.passed)

    @property
    def by_id(self) -> dict[str, ProbeOutcome]:
        return {o.probe_id: o for o in self.outcomes}

    def to_dict(self, *, include_outcomes: bool = True) -> dict:
        payload = {
            "run_id": self.run_id,
            "label": self.label,
            "dataset_id": self.dataset_id,
            "manifest_id": self.manifest_id,
            "config_fingerprint": self.config_fingerprint,
            "k": self.k,
            "notes": self.notes,
            "environment": dict(self.environment),
            "aggregate": self.aggregate.to_dict(),
            "failure_count": len(self.failures),
        }
        if include_outcomes:
            payload["outcomes"] = [o.to_dict() for o in self.outcomes]
        return payload

    def save(self, path) -> str:
        target = str(path)
        directory = os.path.dirname(target)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return target

    @classmethod
    def load(cls, path) -> "RunReport":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        aggregate_payload = payload["aggregate"]
        outcomes = tuple(
            ProbeOutcome(
                probe_id=o["probe_id"], kind=o["kind"], query=o.get("query", ""),
                retrieved_sources=tuple(o.get("retrieved_sources", ())),
                context_text="", recall_hit=o["recall_hit"], facts_hit=o["facts_hit"],
                clean=o["clean"],
                precedence_ok=o.get("precedence_ok", True),
                authz_violation=o.get("authz_violation", False),
                preference_violation=o.get("preference_violation", False),
                abstained=o["abstained"], passed=o["passed"],
                first_required_rank=o.get("first_required_rank"),
                missing_sources=tuple(o.get("missing_sources", ())),
                missing_facts=tuple(o.get("missing_facts", ())),
                leaked_sources=tuple(o.get("leaked_sources", ())),
                latency_ms=float(o.get("latency_ms", 0.0)),
                candidates_considered=int(o.get("candidates_considered", 0)),
                confidence=float(o.get("confidence", 0.0)),
                escalations=int(o.get("escalations", 0)),
                strategy=o.get("strategy", ""), error=o.get("error", ""),
            )
            for o in payload.get("outcomes", ())
        )
        return cls(
            run_id=payload["run_id"], label=payload.get("label", ""),
            dataset_id=payload.get("dataset_id", ""), manifest_id=payload.get("manifest_id", ""),
            config_fingerprint=payload.get("config_fingerprint", ""),
            aggregate=AggregateReport(
                probe_count=aggregate_payload["probe_count"],
                pass_rate=aggregate_payload["pass_rate"],
                recall_at_k=aggregate_payload.get(f"recall_at_{payload.get('k', 5)}", 0.0),
                context_precision=aggregate_payload.get("context_precision", 0.0),
                ndcg_at_k=aggregate_payload.get(f"ndcg_at_{payload.get('k', 5)}", 0.0),
                mrr=aggregate_payload.get("mrr", 0.0),
                fact_coverage=aggregate_payload.get("fact_coverage", 0.0),
                abstention_accuracy=aggregate_payload.get("abstention_accuracy", 0.0),
                authz_violations=int(aggregate_payload.get("authz_violations", 0)),
                preference_violations=int(aggregate_payload.get("preference_violations", 0)),
                leakage_count=int(aggregate_payload.get("leakage_count", 0)),
                mean_latency_ms=aggregate_payload.get("mean_latency_ms", 0.0),
                p95_latency_ms=aggregate_payload.get("p95_latency_ms", 0.0),
                mean_confidence=aggregate_payload.get("mean_confidence", 0.0),
                mean_escalations=aggregate_payload.get("mean_escalations", 0.0),
                k=int(payload.get("k", 5)),
                by_kind=aggregate_payload.get("by_kind", {}),
            ),
            outcomes=outcomes, k=int(payload.get("k", 5)),
            environment=payload.get("environment", {}), notes=payload.get("notes", ""),
        )

    def to_markdown(self, *, max_failures: int = 25) -> str:
        """Human-readable report.

        Ordered so the actionable line is first: what broke, and which kind of
        failure it was. A reader deciding whether to roll back should not have to
        scroll past a table of passing probes.
        """
        agg = self.aggregate
        lines = [
            f"# Evaluation: {self.label or self.run_id}",
            "",
            f"- run: `{self.run_id}`",
            f"- dataset: `{self.dataset_id}`  corpus: `{self.manifest_id}`  config: `{self.config_fingerprint}`",
            f"- probes: {agg.probe_count}   k: {self.k}",
            f"- **pass rate: {agg.pass_rate:.3f}** ({len(self.failures)} failing)",
            "",
            "| metric | value |",
            "| --- | --- |",
            f"| recall@{self.k} | {agg.recall_at_k:.3f} |",
            f"| nDCG@{self.k} | {agg.ndcg_at_k:.3f} |",
            f"| MRR | {agg.mrr:.3f} |",
            f"| context precision | {agg.context_precision:.3f} |",
            f"| fact coverage | {agg.fact_coverage:.3f} |",
            f"| abstention accuracy | {agg.abstention_accuracy:.3f} |",
            f"| **authz violations** | **{agg.authz_violations}** |",
            f"| preference violations | {agg.preference_violations} |",
            f"| leakage events (total) | {agg.leakage_count} |",
            f"| mean / p95 latency | {agg.mean_latency_ms:.1f} / {agg.p95_latency_ms:.1f} ms |",
            "",
            "## By failure mode",
            "",
            "| kind | probes | pass rate |",
            "| --- | --- | --- |",
        ]
        for kind, values in agg.by_kind.items():
            lines.append(
                f"| {kind} | {int(values.get('probe_count', 0))} | {values.get('pass_rate', 1.0):.3f} |"
            )

        if self.failures:
            lines += ["", "## Failures", ""]
            for outcome in self.failures[:max_failures]:
                reasons = []
                if outcome.leaked_sources:
                    reasons.append(f"leaked {list(outcome.leaked_sources)}")
                if outcome.missing_sources:
                    reasons.append(f"missing {list(outcome.missing_sources)}")
                if outcome.missing_facts:
                    reasons.append(f"facts absent {list(outcome.missing_facts)[:3]}")
                if not outcome.precedence_ok and not outcome.leaked_sources:
                    reasons.append("required document did not outrank the distractor")
                if outcome.kind == "unanswerable" and not outcome.abstained:
                    reasons.append("did not abstain")
                if outcome.error:
                    reasons.append(f"error: {outcome.error}")
                if not reasons:
                    reasons.append("contract unmet")
                rank = f"rank {outcome.first_required_rank}" if outcome.first_required_rank else "not retrieved"
                lines.append(
                    f"- **{outcome.probe_id}** ({outcome.kind}, {rank}) - {'; '.join(reasons)}"
                )
            if len(self.failures) > max_failures:
                lines.append(f"- ...and {len(self.failures) - max_failures} more")
        return "\n".join(lines) + "\n"


def run_probe_set(
    probes: ProbeSet,
    executor: ProbeExecutor,
    *,
    label: str = "",
    manifest_id: str = "",
    config_fingerprint_value: str = "",
    k: int = 5,
    notes: str = "",
) -> RunReport:
    """Execute every probe and return a report.

    Sequential by design. Probes are cheap, the corpus is in memory, and
    parallel execution would make latency figures meaningless -- the number this
    system is judged on most often after accuracy, and the one most easily
    destroyed by an unrelated thread competing for the GIL.
    """
    outcomes: list[ProbeOutcome] = []
    for probe in probes:
        execution = executor(probe)
        outcomes.append(
            evaluate_probe(
                probe,
                execution.ranked_sources,
                execution.context_text,
                abstained=execution.abstained,
                k=k,
                latency_ms=execution.latency_ms,
                candidates_considered=execution.candidates_considered,
                confidence=execution.confidence,
                escalations=execution.escalations,
                strategy=execution.strategy,
            )
        )

    material = "|".join(
        [
            probes.dataset_id or probes.name,
            manifest_id,
            config_fingerprint_value,
            ",".join(f"{o.probe_id}:{int(o.passed)}" for o in outcomes),
        ]
    )
    run_id = hashlib.sha256(material.encode()).hexdigest()[:12]

    return RunReport(
        run_id=run_id,
        label=label or probes.name,
        dataset_id=probes.dataset_id,
        manifest_id=manifest_id,
        config_fingerprint=config_fingerprint_value,
        aggregate=aggregate(outcomes, probes.probes, k=k),
        outcomes=tuple(outcomes),
        k=k,
        environment={
            "python": sys.version.split()[0],
            "platform": platform.platform(terse=True),
            "executable": os.path.basename(sys.executable),
        },
        notes=notes,
    )
