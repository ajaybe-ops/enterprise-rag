"""Deterministic retrieval metrics.

Everything in this module is a pure function with no model call, no randomness,
and no network. That is a hard requirement, not a stylistic preference:

    A gate that can vary run to run cannot be a gate.

If recall@5 is 0.82 on Monday and 0.84 on Tuesday for identical inputs, nobody
can tell whether the code changed or the sampler did, so the number stops being
evidence. Everything here is reproducible from the inputs alone.

Graded metrics are computed on *document* rank, not chunk rank. An enterprise
question is answered by a document, and a document that has been chunked into
six pieces will occupy six adjacent ranks in a chunk-level ranking -- inflating
recall for free. Collapsing to first-rank-per-source removes that advantage, so
a probe cannot be passed merely by having a long document.

What is deliberately absent: any embedding-based or LLM-based "semantic
relevance" score. It is non-deterministic, opaque, and -- most importantly --
cannot be used to decide whether a change is safe to ship. Judgement-based
scoring belongs in a separate, non-gating suite (docs/adr/0003).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from enterprise_rag.evaluation.probe import Probe, ProbeOutcome

__all__ = [
    "AggregateReport",
    "aggregate",
    "context_precision",
    "evaluate_probe",
    "fact_coverage",
    "first_required_rank",
    "leakage",
    "ndcg_at_k",
    "normalise",
    "precedence_satisfied",
    "recall_at_k",
    "reciprocal_rank",
    "source_recall",
]

_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Casefold and collapse whitespace for literal fact matching.

    Not lowercasing is the one shortcut that would have made this gate pass
    documents that differ from the probe in a way a reader would call a
    different value. Casefolding plus whitespace collapse fixes the two
    transformations that genuinely do not change meaning -- capitalisation and
    line wrapping -- and nothing else.
    """
    return _WS.sub(" ", text).strip().casefold()


# -- ranking primitives ---------------------------------------------------- #


def first_required_rank(ranked_sources: Sequence[str], required: Iterable[str]) -> int | None:
    """1-based rank of the earliest required source, or None if absent."""
    wanted = set(required)
    for position, source in enumerate(ranked_sources, start=1):
        if source in wanted:
            return position
    return None


def source_recall(retrieved_sources: Iterable[str], required: Iterable[str]) -> float:
    """Fraction of required *sources* that appear at any rank.

    1.0 for a negative probe with no requirements, which makes the aggregate
    score comparable across mixed positive/negative suites instead of silently
    penalising the negative half.
    """
    wanted = set(required)
    if not wanted:
        return 1.0
    return len(wanted & set(retrieved_sources)) / len(wanted)


def recall_at_k(ranked_sources: Sequence[str], required: Iterable[str], k: int) -> float:
    """Fraction of required sources present within the top `k` ranks."""
    return source_recall(ranked_sources[:k], required)


def context_precision(retrieved_sources: Sequence[str], required: Iterable[str]) -> float:
    """How much of what was retrieved was actually asked for.

    Rewards a retriever for returning *less* when the question only needed one
    document. The symmetric cost -- retrieving a distractor and letting the model
    read it -- is precisely the failure that produces confident, well-cited,
    wrong answers, so precision is tracked separately from recall rather than
    folded into an F1 that hides the trade-off.
    """
    if not retrieved_sources:
        return 0.0
    wanted = set(required)
    return len(wanted & set(retrieved_sources)) / len(retrieved_sources)


def reciprocal_rank(ranked_sources: Sequence[str], required: Iterable[str]) -> float:
    """1/rank of the first required source; 0.0 if never retrieved.

    The headline number for "how far from the top is the answer", which is what
    determines whether a reranker downstream ever gets a chance to help.
    """
    rank = first_required_rank(ranked_sources, required)
    return 0.0 if rank is None else 1.0 / rank


def ndcg_at_k(ranked_sources: Sequence[str], required: Iterable[str], k: int) -> float:
    """Binary-gain nDCG@k, the graded companion to recall@k.

    Recall@5 cannot distinguish "required docs at ranks 1 and 2" from "required
    docs at ranks 4 and 5". nDCG can, which is what makes it the metric to
    optimise when the retriever's only lever is ordering.
    """
    wanted = set(required)
    if not wanted:
        return 1.0

    def dcg(sequence: Sequence[str]) -> float:
        return sum(
            1.0 / math.log2(position + 1)
            for position, source in enumerate(sequence, start=1)
            if source in wanted
        )

    actual = dcg(ranked_sources[:k])
    ideal_count = min(len(wanted), k)
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_count + 1))
    return actual / ideal if ideal else 0.0


def fact_coverage(context: str, must_contain: Sequence[str]) -> float:
    """Fraction of required facts present in the assembled context.

    This is the check that source-level recall cannot make. A retriever can hit
    100% recall on a document and still hand the model a context window that
    excludes the paragraph containing the answer, because the document was long
    and the matching sentence fell outside the retrieved spans.
    """
    if not must_contain:
        return 1.0
    haystack = normalise(context)
    hits = sum(1 for fact in must_contain if normalise(fact) in haystack)
    return hits / len(must_contain)


def leakage(retrieved_sources: Sequence[str], must_not: Sequence[str]) -> tuple[str, ...]:
    """Forbidden sources that surfaced. Empty is the only acceptable value."""
    forbidden = set(must_not)
    seen: list[str] = []
    for source in retrieved_sources:
        if source in forbidden and source not in seen:
            seen.append(source)
    return tuple(seen)


def precedence_satisfied(
    ranked_sources: Sequence[str], must_outrank: Sequence[str], must_not: Sequence[str]
) -> bool:
    """Whether every `must_outrank` document outranks every `must_not` document.

    Unranked documents are treated as losing. A `must_outrank` document that was
    not retrieved at all has not outranked anything, and crediting it would mean
    a probe passes on a result set that contains neither document -- which is the
    same vacuous pass that a bare set-membership check is vulnerable to.
    """
    if not must_outrank:
        return True
    blockers = set(must_not)
    positions = {source: index for index, source in enumerate(ranked_sources, start=1)}

    best_blocker = min(
        (positions[s] for s in blockers if s in positions), default=None
    )
    if best_blocker is None:
        # Nothing to outrank is present, so every required precedence holds.
        return all(s in positions for s in must_outrank)

    return all(
        s in positions and positions[s] < best_blocker for s in must_outrank
    )


# -- per-probe evaluation -------------------------------------------------- #


def evaluate_probe(
    probe: Probe,
    ranked_sources: Sequence[str],
    context_text: str = "",
    *,
    abstained: bool = False,
    k: int = 5,
    latency_ms: float = 0.0,
    candidates_considered: int = 0,
    confidence: float = 0.0,
    escalations: int = 0,
    strategy: str = "",
) -> ProbeOutcome:
    """Score one probe against one retrieval result.

    `passed` is the conjunction of every applicable obligation.

    For a negative probe (`should_abstain`) that means: no forbidden source
    surfaced, *and* the system abstained. Requiring both is deliberate --
    retrieving a document you should not have, then correctly declining to quote
    it, is still a retrieval failure, because the sensitive text was in the
    model's context either way.

    For an answerable probe it means: the required sources were retrieved, the
    required facts are in the context, precedence holds, nothing leaked, *and*
    the system actually answered.

    That last clause was missing, and it is the more consequential half. Without
    it `passed` was computed purely from the retrieval result and the context
    text, so a probe whose retrieved evidence was perfect still passed when the
    pipeline declined to answer it. That flatters the system in the one
    direction that matters: it converts "answered wrongly" or "refused a
    question it should have been able to answer" into a pass, because neither
    shows up in the retrieval metrics. Nine multi-hop probes were abstaining at
    confidence 0.2558 with the correct documents already retrieved, and the
    suite reported them as failures for the wrong reason -- `missing_sources`
    was blamed when the real defect was the pipeline declining to answer.

    Abstention on an answerable probe is a failure of the *system*, not of
    retrieval, and it has to be scored as one.
    """
    rank = first_required_rank(ranked_sources, probe.must_retrieve)
    missing_sources = tuple(s for s in probe.must_retrieve if s not in set(ranked_sources))
    missing_facts = tuple(f for f in probe.must_contain if normalise(f) not in normalise(context_text))
    leaked = leakage(ranked_sources, probe.must_not)

    recall_hit = recall_at_k(ranked_sources, probe.must_retrieve, k) >= 1.0
    facts_hit = fact_coverage(context_text, probe.must_contain) >= 1.0

    # `must_not` means two different things depending on whether a precedence
    # obligation is attached, and conflating them is the mistake that made the
    # first version of this suite fail a correct system:
    #
    #   with must_outrank -> `must_not` is the *loser* set. Its presence is
    #                       allowed; being ranked above the winner is not. So
    #                       `clean` must not be derived from it, or a retriever
    #                       that correctly put the draft first and the SLO second
    #                       is reported as a violation.
    #   without          -> `must_not` is an absolute exclusion and `clean` is
    #                       the obligation.
    #
    # Either way, an *authorisation* must_not is absolute regardless: a document
    # the caller may not read is a disclosure at any rank.
    precedence_ok = precedence_satisfied(
        ranked_sources, probe.must_outrank, probe.must_not
    )
    is_authorization = probe.must_not_kind == "authorization"
    if is_authorization or not probe.must_outrank:
        clean = not leaked
    else:
        clean = True

    authz_violation = is_authorization and bool(leaked)
    preference_violation = (not is_authorization) and bool(probe.must_outrank) and not precedence_ok

    if probe.should_abstain:
        passed = clean and abstained
    else:
        # An answerable probe also requires an answer. See the docstring: without
        # `not abstained` here, a pipeline that retrieved perfectly and then
        # refused to answer was scored as a pass.
        passed = recall_hit and facts_hit and clean and precedence_ok and not abstained

    return ProbeOutcome(
        probe_id=probe.probe_id,
        kind=probe.kind,
        query=probe.query,
        retrieved_sources=tuple(ranked_sources),
        context_text=context_text,
        recall_hit=recall_hit,
        facts_hit=facts_hit,
        clean=clean,
        precedence_ok=precedence_ok,
        authz_violation=authz_violation,
        preference_violation=preference_violation,
        abstained=abstained,
        passed=passed,
        first_required_rank=rank,
        missing_sources=missing_sources,
        missing_facts=missing_facts,
        leaked_sources=leaked,
        latency_ms=latency_ms,
        candidates_considered=candidates_considered,
        confidence=confidence,
        escalations=escalations,
        strategy=strategy,
    )


# -- aggregation ----------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class AggregateReport:
    """Suite-level metrics, with a per-kind breakdown.

    The per-kind breakdown is not decoration. A single blended pass rate hides
    the failure that matters: a suite can hold steady while every
    `version_conflict` probe starts failing, because that is a small slice of a
    suite that is mostly easy. Since a superseded-document regression is the most
    dangerous kind for a deployment, it must be impossible to miss in a report.
    """

    probe_count: int
    pass_rate: float
    recall_at_k: float
    context_precision: float
    ndcg_at_k: float
    mrr: float
    fact_coverage: float
    abstention_accuracy: float
    authz_violations: int
    preference_violations: int
    leakage_count: int
    mean_latency_ms: float
    p95_latency_ms: float
    mean_confidence: float
    mean_escalations: float
    k: int
    by_kind: dict[str, dict[str, float]]

    def to_dict(self) -> dict:
        return {
            "probe_count": self.probe_count,
            "pass_rate": round(self.pass_rate, 4),
            f"recall_at_{self.k}": round(self.recall_at_k, 4),
            "context_precision": round(self.context_precision, 4),
            f"ndcg_at_{self.k}": round(self.ndcg_at_k, 4),
            "mrr": round(self.mrr, 4),
            "fact_coverage": round(self.fact_coverage, 4),
            "abstention_accuracy": round(self.abstention_accuracy, 4),
            "authz_violations": self.authz_violations,
            "preference_violations": self.preference_violations,
            "leakage_count": self.leakage_count,
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "mean_confidence": round(self.mean_confidence, 4),
            "mean_escalations": round(self.mean_escalations, 3),
            "k": self.k,
            "by_kind": {
                kind: {name: round(value, 4) for name, value in values.items()}
                for kind, values in self.by_kind.items()
            },
        }

    def weakest_kinds(self, limit: int = 3) -> list[tuple[str, float]]:
        """Kinds with the lowest pass rate, worst first."""
        scored = [
            (kind, values.get("pass_rate", 1.0))
            for kind, values in self.by_kind.items()
            if values.get("probe_count", 0) > 0
        ]
        return sorted(scored, key=lambda item: item[1])[:limit]


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile. No interpolation, so a p95 over 20 probes is an
    actual observed latency rather than a synthetic one between two samples."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def aggregate(
    outcomes: Sequence[ProbeOutcome],
    probes: Sequence[Probe] = (),
    *,
    k: int = 5,
) -> AggregateReport:
    """Reduce per-probe outcomes to suite metrics.

    An empty suite returns zeros rather than raising. A run that measured nothing
    is a fact worth reporting, and fabricating a 1.0 for "recall of no probes"
    would turn an instrumentation bug into an impressive green dashboard.
    """
    if not outcomes:
        return AggregateReport(
            probe_count=0, pass_rate=0.0, recall_at_k=0.0, context_precision=0.0,
            ndcg_at_k=0.0, mrr=0.0, fact_coverage=0.0, abstention_accuracy=0.0,
            authz_violations=0, preference_violations=0, leakage_count=0,
            mean_latency_ms=0.0, p95_latency_ms=0.0,
            mean_confidence=0.0, mean_escalations=0.0, k=k, by_kind={},
        )

    by_id = {p.probe_id: p for p in probes}
    total = len(outcomes)

    def mrr_of(outcome: ProbeOutcome) -> float:
        probe = by_id.get(outcome.probe_id)
        return reciprocal_rank(outcome.retrieved_sources, probe.must_retrieve) if probe else 0.0

    abstain_probes = [o for o in outcomes if (by_id.get(o.probe_id) and by_id[o.probe_id].should_abstain)]
    latencies = [o.latency_ms for o in outcomes]

    kinds: dict[str, list[ProbeOutcome]] = {}
    for outcome in outcomes:
        kinds.setdefault(outcome.kind, []).append(outcome)

    by_kind: dict[str, dict[str, float]] = {}
    for kind, group in sorted(kinds.items()):
        group_probes = [by_id[o.probe_id] for o in group if o.probe_id in by_id]
        by_kind[kind] = {
            "probe_count": float(len(group)),
            "pass_rate": sum(1 for o in group if o.passed) / len(group),
            "recall_at_k": sum(
                recall_at_k(o.retrieved_sources, p.must_retrieve, k)
                for o, p in zip(group, group_probes, strict=False)
            ) / len(group) if len(group_probes) == len(group) else 0.0,
            "fact_coverage": sum(
                fact_coverage(o.context_text, p.must_contain)
                for o, p in zip(group, group_probes, strict=False)
            ) / len(group) if len(group_probes) == len(group) else 0.0,
            "authz_violations": float(sum(1 for o in group if o.authz_violation)),
            "preference_violations": float(sum(1 for o in group if o.preference_violation)),
        }

    return AggregateReport(
        probe_count=total,
        pass_rate=sum(1 for o in outcomes if o.passed) / total,
        recall_at_k=sum(
            recall_at_k(o.retrieved_sources, by_id[o.probe_id].must_retrieve, k)
            for o in outcomes
            if o.probe_id in by_id
        ) / max(1, sum(1 for o in outcomes if o.probe_id in by_id)),
        context_precision=sum(
            context_precision(o.retrieved_sources, by_id[o.probe_id].must_retrieve)
            for o in outcomes
            if o.probe_id in by_id
        ) / max(1, sum(1 for o in outcomes if o.probe_id in by_id)),
        ndcg_at_k=sum(
            ndcg_at_k(o.retrieved_sources, by_id[o.probe_id].must_retrieve, k)
            for o in outcomes
            if o.probe_id in by_id
        ) / max(1, sum(1 for o in outcomes if o.probe_id in by_id)),
        mrr=sum(mrr_of(o) for o in outcomes) / total,
        fact_coverage=sum(
            fact_coverage(o.context_text, by_id[o.probe_id].must_contain)
            for o in outcomes
            if o.probe_id in by_id
        ) / max(1, sum(1 for o in outcomes if o.probe_id in by_id)),
        abstention_accuracy=(
            sum(1 for o in abstain_probes if o.abstained) / len(abstain_probes)
            if abstain_probes
            else 1.0
        ),
        authz_violations=sum(1 for o in outcomes if o.authz_violation),
        preference_violations=sum(1 for o in outcomes if o.preference_violation),
        leakage_count=sum(len(o.leaked_sources) for o in outcomes),
        mean_latency_ms=sum(latencies) / total,
        p95_latency_ms=_percentile(latencies, 0.95),
        mean_confidence=sum(o.confidence for o in outcomes) / total,
        mean_escalations=sum(o.escalations for o in outcomes) / total,
        k=k,
        by_kind=by_kind,
    )
