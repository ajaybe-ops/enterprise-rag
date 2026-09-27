"""Regression detection with causal attribution.

The problem
-----------
"Recall@5 dropped from 0.91 to 0.84" is not actionable. Four different root
causes produce exactly that line, and the correct response to each is different:

    corpus edit      -> the document changed; fix or re-chunk the document
    re-chunk         -> the text is identical; the chunker changed; tune boundaries
    new distractor   -> a new document now outranks the right one; curate or scope
    code regression  -> the pipeline changed; the corpus is innocent

Teams overwhelmingly investigate in the wrong order, because the report gives
them no way to tell the cases apart and the corpus is the thing that changed most
recently. The diff data needed to disambiguate already exists -- it is just
usually thrown away at ingest time and never reconstructed afterwards.

This module reconstructs it. Given a baseline run, a candidate run, and the two
corpus manifests, it attributes every per-probe regression to a cause, ranks the
candidate causes by their supporting evidence, and reports what it could *not*
explain.

The honesty rule
----------------
`RegressionAttribution` always carries a `Cause` with `confidence`, and the
attributor emits `unattributed` rather than guessing when the evidence does not
support a specific cause. An attribution engine that always produces a confident
answer is worse than none: it converts a real unknown into a false certainty
that people then act on. Every claim here is traceable to a specific recorded
fact -- a changed source, a fingerprint delta, a rank movement -- and where that
is not possible the verdict is `unattributed`.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from enterprise_rag.evaluation.probe import Probe, ProbeOutcome, ProbeSet
from enterprise_rag.evaluation.runner import RunReport
from enterprise_rag.indexing.manifest import CorpusManifest, ManifestDiff

__all__ = [
    "Attribution",
    "Cause",
    "ProbeRegression",
    "RegressionReport",
    "attribute",
    "compare",
]

CauseKind = Literal[
    "unattributed",
    "corpus_edit",
    "corpus_removal",
    "corpus_addition",
    "rechunk_shift",
    "config_change",
    "code_regression",
    "dataset_change",
]


@dataclass(frozen=True, slots=True)
class Cause:
    """One attributed cause plus the evidence that supports it.

    `evidence` is a list of concrete facts, not a rationale sentence. A reader
    must be able to check each item against the two manifests themselves; if the
    attribution cannot show its work it does not deserve to be believed.
    """

    kind: CauseKind
    summary: str
    confidence: float
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "summary": self.summary,
            "confidence": round(self.confidence, 3),
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class ProbeRegression:
    """A probe that passed on the baseline and fails on the candidate."""

    probe_id: str
    kind: str
    query: str
    was_rank: int | None
    now_rank: int | None
    rank_delta: int | None
    obligations_lost: tuple[str, ...]
    obligations_gained: tuple[str, ...]
    authz_lost: bool = False
    cause: Cause = None  # type: ignore[assignment]
    alternatives: tuple[Cause, ...] = ()

    @property
    def severity(self) -> str:
        """Ordering for triage, and the release gate.

        An authorisation violation is an incident; a lost ranking is a metric.
        Ranking them identically by pass/fail is how a security regression gets
        deprioritised behind a noisy recall wobble, so the two are separated at
        the point where the decision is actually made.
        """
        if self.authz_lost:
            return "critical"
        if "facts_hit" in self.obligations_lost:
            return "high"
        if "abstained" in self.obligations_lost:
            return "high"
        return "medium"

    def to_dict(self) -> dict:
        return {
            "probe_id": self.probe_id,
            "kind": self.kind,
            "query": self.query,
            "severity": self.severity,
            "was_rank": self.was_rank,
            "now_rank": self.now_rank,
            "rank_delta": self.rank_delta,
            "obligations_lost": list(self.obligations_lost),
            "obligations_gained": list(self.obligations_gained),
            "authz_lost": self.authz_lost,
            "cause": self.cause.to_dict(),
            "alternatives": [c.to_dict() for c in self.alternatives],
        }


@dataclass(frozen=True, slots=True)
class Attribution:
    """Suite-level verdict: what kind of change was this?"""

    verdict: CauseKind
    summary: str
    evidence: tuple[str, ...] = ()
    corpus_changed: bool = False
    config_changed: bool = False
    dataset_changed: bool = False

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "summary": self.summary,
            "evidence": list(self.evidence),
            "corpus_changed": self.corpus_changed,
            "config_changed": self.config_changed,
            "dataset_changed": self.dataset_changed,
        }


@dataclass(frozen=True, slots=True)
class MetricDelta:
    name: str
    baseline: float
    candidate: float
    tolerance: float = 0.0
    higher_is_better: bool = True

    @property
    def delta(self) -> float:
        return self.candidate - self.baseline

    @property
    def regressed(self) -> bool:
        """Direction-aware, which is the only correct way to compare a metric.

        The previous implementation returned `delta < -tolerance` for every
        metric, with a docstring claiming "lower is better for every metric
        tracked here, including latency". Both halves of that are wrong: recall
        and pass rate are better *higher*, and latency and leak counts are worse
        *higher*. The consequence was not a cosmetic mislabel -- a pipeline that
        became 10x slower, or that started leaking forbidden documents, moved
        its metric in the un-flagged direction and was reported as healthy. The
        gate silently protected against exactly the failures it was built to
        catch.

        `tolerance` is applied to the magnitude of a worsening change in either
        direction, because a deterministic pipeline still shows sub-millisecond
        latency jitter and a gate that fires on machine noise teaches people to
        ignore it.
        """
        if self.higher_is_better:
            return self.delta < -abs(self.tolerance)
        return self.delta > abs(self.tolerance)

    def to_dict(self) -> dict:
        return {
            "metric": self.name,
            "baseline": round(self.baseline, 4),
            "candidate": round(self.candidate, 4),
            "delta": round(self.delta, 4),
            "higher_is_better": self.higher_is_better,
            "regressed": self.regressed,
        }


@dataclass(frozen=True, slots=True)
class RegressionReport:
    baseline: RunReport
    candidate: RunReport
    attribution: Attribution
    manifest_diff: ManifestDiff | None
    regressions: tuple[ProbeRegression, ...] = ()
    improvements: tuple[str, ...] = ()
    metric_deltas: tuple[MetricDelta, ...] = ()
    per_kind_deltas: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    @property
    def is_comparable(self) -> bool:
        """False when the two runs did not measure the same thing.

        Comparing across datasets produces a number that is arithmetically valid
        and completely meaningless, so it is refused rather than reported.
        """
        return not self.attribution.dataset_changed

    @property
    def blocking(self) -> tuple[ProbeRegression, ...]:
        """Regressions severe enough to block a release on their own."""
        return tuple(r for r in self.regressions if r.severity in {"critical", "high"})

    def to_dict(self) -> dict:
        return {
            "baseline_run": self.baseline.run_id,
            "candidate_run": self.candidate.run_id,
            "is_comparable": self.is_comparable,
            "attribution": self.attribution.to_dict(),
            "corpus_diff": self.manifest_diff.to_dict() if self.manifest_diff else None,
            "metric_deltas": [d.to_dict() for d in self.metric_deltas],
            "per_kind_deltas": {
                kind: {name: round(value, 4) for name, value in values.items()}
                for kind, values in self.per_kind_deltas.items()
            },
            "regressions": [r.to_dict() for r in self.regressions],
            "improvements": list(self.improvements),
            "blocking_count": len(self.blocking),
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Regression report: {self.baseline.run_id} -> {self.candidate.run_id}",
            "",
            f"**Verdict: {self.attribution.verdict}** - {self.attribution.summary}",
            "",
        ]
        if self.attribution.evidence:
            lines += ["Evidence:"]
            lines += [f"- {item}" for item in self.attribution.evidence]
            lines.append("")

        if not self.is_comparable:
            lines += [
                "> The two runs used different probe sets, so metric deltas are not "
                "reported. Re-run both against the same dataset to compare.",
                "",
            ]

        if self.metric_deltas:
            lines += [
                "| metric | baseline | candidate | delta | regressed |",
                "| --- | --- | --- | --- | --- |",
            ]
            lines += [
                f"| {d.name} | {d.baseline:.4f} | {d.candidate:.4f} | {d.delta:+.4f} | "
                f"{'YES' if d.regressed else 'no'} |"
                for d in self.metric_deltas
            ]
            lines.append("")

        if self.manifest_diff and not self.manifest_diff.is_empty:
            lines += [f"Corpus: {self.manifest_diff.summary()}", ""]
            for change in (
                *self.manifest_diff.added,
                *self.manifest_diff.removed,
                *self.manifest_diff.modified,
            ):
                lines.append(f"- {change.explain()}")
            lines.append("")

        if self.regressions:
            lines += ["## Regressions", ""]
            for regression in sorted(self.regressions, key=lambda r: (r.severity, r.probe_id)):
                lines.append(
                    f"### [{regression.severity}] {regression.probe_id} ({regression.kind})"
                )
                lines.append(f"- query: {regression.query}")
                movement = (
                    "not retrieved"
                    if regression.now_rank is None
                    else f"rank {regression.was_rank} -> {regression.now_rank}"
                    if regression.was_rank
                    else f"rank -> {regression.now_rank}"
                )
                lines.append(f"- {movement}")
                if regression.obligations_lost:
                    lines.append(f"- obligations lost: {', '.join(regression.obligations_lost)}")
                if regression.obligations_gained:
                    lines.append(f"- obligations regained: {', '.join(regression.obligations_gained)}")
                lines.append(f"- **cause: {regression.cause.kind}** - {regression.cause.summary}")
                lines += [f"  - {item}" for item in regression.cause.evidence]
                if regression.alternatives:
                    lines.append(
                        "- other candidates: "
                        + "; ".join(f"{c.kind} ({c.confidence:.2f})" for c in regression.alternatives)
                    )
                lines.append("")
        else:
            lines += ["No per-probe regressions.", ""]

        if self.improvements:
            lines += [f"Improved: {', '.join(self.improvements)}", ""]

        return "\n".join(lines)


# -- attribution ----------------------------------------------------------- #


def _attribute_probe(
    probe: Probe,
    before: ProbeOutcome,
    after: ProbeOutcome,
    diff: ManifestDiff | None,
    config_changed: bool,
) -> ProbeRegression:
    """Attribute one probe's regression to a cause, with alternatives."""
    was_rank = getattr(before, "first_required_rank", None)
    now_rank = getattr(after, "first_required_rank", None)
    rank_delta = (
        None if was_rank is None or now_rank is None else now_rank - was_rank
    )

    lost = tuple(
        name
        for name, was, now in (
            ("recall_hit", before.recall_hit, after.recall_hit),
            ("facts_hit", before.facts_hit, after.facts_hit),
            ("clean", before.clean, after.clean),
            ("abstained", before.abstained, after.abstained),
        )
        if was and not now
    )
    gained = tuple(
        name
        for name, was, now in (
            ("recall_hit", before.recall_hit, after.recall_hit),
            ("facts_hit", before.facts_hit, after.facts_hit),
            ("clean", before.clean, after.clean),
            ("abstained", before.abstained, after.abstained),
        )
        if not was and now
    )

    # An access-control regression is a different event from a ranking
    # regression, and is tracked separately so it can gate the release on its
    # own. Derived from the probe's declared severity rather than from whether
    # the run happened to record an `authz_violation` flag, so it is still
    # correct for a report written before that flag existed.
    authz_lost = probe.must_not_kind == "authorization" and before.clean and not after.clean

    # Candidate causes, each with its own evidence, ranked by specificity. The
    # order encodes a prior: a concrete corpus event explains more than a
    # configuration change, which explains more than "the code changed".
    candidates: list[Cause] = []

    if diff is not None:
        for change in diff.removed:
            if change.source in probe.must_retrieve:
                candidates.append(
                    Cause(
                        kind="corpus_removal",
                        summary=f"required document {change.source} was deleted from the corpus",
                        confidence=0.99,
                        evidence=(
                            f"manifest diff: {change.explain()}",
                            f"probe {probe.probe_id} lists {change.source} in must_retrieve",
                        ),
                    )
                )
        for change in diff.modified:
            if change.source in probe.must_retrieve:
                # A document in `modified` exists in both manifests, so both
                # records are available. `both()` states that and fails loudly if
                # it is ever not true.
                before_doc, after_doc = change.both()
                if change.rechunked:
                    candidates.append(
                        Cause(
                            kind="rechunk_shift",
                            summary=(
                                f"{change.source} has identical text but a different chunk "
                                f"decomposition ({change.chunk_delta:+d} chunks)"
                            ),
                            confidence=0.9,
                            evidence=(
                                f"content hash unchanged ({after_doc.content_hash})",
                                change.explain(),
                                "rank shift is therefore a boundary effect, not a content effect",
                            ),
                        )
                    )
                else:
                    candidates.append(
                        Cause(
                            kind="corpus_edit",
                            summary=f"required document {change.source} was edited",
                            confidence=0.75,
                            evidence=(
                                f"content {before_doc.content_hash} -> {after_doc.content_hash}",
                                change.explain(),
                            ),
                        )
                    )
        for change in diff.added:
            # A new document is only a cause if it actually displaced the
            # evidence. A new document that appeared and ranked last changed
            # nothing, and claiming otherwise would make every corpus addition
            # look like a possible culprit.
            now_sources = set(getattr(after, "retrieved_sources", ()))
            was_sources = set(getattr(before, "retrieved_sources", ()))
            if change.source in now_sources and change.source not in was_sources:
                required = set(probe.must_retrieve)
                # Parenthesised deliberately: `&` and `-` share precedence in
                # Python and associate left to right, so the unparenthesised
                # form reduces to `(required & was) - required` and is always
                # empty -- which would make every addition look non-displacing.
                displaced = (required & was_sources) - (required & now_sources)
                candidates.append(
                    Cause(
                        kind="corpus_addition",
                        summary=f"new document {change.source} entered the results",
                        confidence=0.8 if displaced else 0.5,
                        evidence=(
                            f"present in candidate results, absent from baseline: {change.source}",
                            change.explain(),
                            (
                                f"displaced: {sorted(displaced)}"
                                if displaced
                                else "no previously-retrieved required document was displaced"
                            ),
                        ),
                    )
                )
        for change in diff.modified:
            if change.source in probe.must_not and "clean" in lost:
                candidates.append(
                    Cause(
                        kind="corpus_edit",
                        summary=f"forbidden document {change.source} was edited and now surfaces",
                        confidence=0.7,
                        evidence=(change.explain(), f"{change.source} is in must_not for {probe.probe_id}"),
                    )
                )

    if config_changed:
        candidates.append(
            Cause(
                kind="config_change",
                summary="retrieval configuration differs between the two runs",
                confidence=0.6 if not candidates else 0.3,
                evidence=(
                    "config fingerprints differ between baseline and candidate",
                ),
            )
        )

    if rank_delta is not None and rank_delta > 0:
        candidates.append(
            Cause(
                kind="code_regression",
                summary=f"required evidence fell {rank_delta} rank(s) with no recorded cause",
                confidence=0.4 if candidates else 0.7,
                evidence=(
                    f"rank {was_rank} -> {now_rank}",
                    "no matching document change in the manifest diff",
                ),
            )
        )

    if not candidates:
        # The honest outcome. Reported as a distinct verdict so that a suite full
        # of unattributed regressions is itself visible as a pattern -- it means
        # the manifest is not capturing whatever actually changed.
        return ProbeRegression(
            probe_id=probe.probe_id,
            kind=probe.kind,
            query=probe.query,
            was_rank=was_rank,
            now_rank=now_rank,
            rank_delta=rank_delta,
            obligations_lost=lost,
            obligations_gained=gained,
            authz_lost=authz_lost,
            cause=Cause(
                kind="unattributed",
                summary="no recorded change explains this regression",
                confidence=0.0,
                evidence=(
                    (
                        "corpus manifest unchanged"
                        if diff is None or diff.is_empty
                        else "corpus changed, but not in a way that touches this probe"
                    ),
                    "configuration unchanged" if not config_changed else "configuration changed",
                    "the pipeline itself is the remaining suspect",
                ),
            ),
        )

    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return ProbeRegression(
        probe_id=probe.probe_id,
        kind=probe.kind,
        query=probe.query,
        was_rank=was_rank,
        now_rank=now_rank,
        rank_delta=rank_delta,
        obligations_lost=lost,
        obligations_gained=gained,
        authz_lost=authz_lost,
        cause=candidates[0],
        alternatives=tuple(candidates[1:]),
    )


def attribute(
    baseline: RunReport,
    candidate: RunReport,
    *,
    probes: ProbeSet | None = None,
    manifest_before: CorpusManifest | None = None,
    manifest_after: CorpusManifest | None = None,
) -> RegressionReport:
    """Compare two runs and attribute every regression to a cause."""
    diff = manifest_before.diff(manifest_after) if manifest_before and manifest_after else None
    dataset_changed = bool(
        baseline.dataset_id and candidate.dataset_id and baseline.dataset_id != candidate.dataset_id
    )
    config_changed = bool(
        baseline.config_fingerprint
        and candidate.config_fingerprint
        and baseline.config_fingerprint != candidate.config_fingerprint
    )
    corpus_changed = bool(diff and not diff.is_empty)

    # -- suite-level verdict ------------------------------------------------ #
    evidence: list[str] = []
    # The corpus branches below repeat `diff is not None` alongside
    # `corpus_changed`. That is not a redundant belt-and-braces check: a type
    # checker cannot infer that `corpus_changed` implies a non-None `diff`, and
    # each of those branches dereferences it. Writing the condition out makes
    # the narrowing visible in the same place the value is used.
    #
    # `verdict` is annotated rather than left to inference because every branch
    # assigns a pair and the first assignment would otherwise pin it to `str`,
    # which is not `CauseKind` -- the closed set the report schema and the
    # consumers of `verdict` both rely on. Declaring it here means a typo in any
    # one branch is caught at the branch rather than surfacing as an unlabelled
    # verdict in a saved report.
    verdict: CauseKind
    summary: str
    if dataset_changed:
        verdict, summary = (
            "dataset_change",
            f"probe sets differ ({baseline.dataset_id} -> {candidate.dataset_id}); "
            "the runs are not comparable",
        )
        evidence.append("re-run both runs against a single probe set before comparing")
    elif corpus_changed and config_changed and diff is not None:
        verdict, summary = (
            "corpus_edit",
            "both the corpus and the configuration changed; isolate them before acting",
        )
        evidence.append(f"corpus: {diff.summary()}")
        evidence.append("config fingerprints differ")
    elif corpus_changed and diff is not None:
        rechunks = diff.rechunked
        if rechunks and not (diff.added or diff.removed):
            verdict = "rechunk_shift"
            summary = f"corpus text is unchanged but {len(rechunks)} document(s) were re-chunked"
        else:
            verdict, summary = "corpus_edit", f"corpus changed ({diff.summary()}) while configuration held"
        evidence.append(f"corpus: {diff.summary()}")
    elif config_changed:
        verdict, summary = "config_change", "configuration changed while corpus and probes held"
    else:
        verdict, summary = (
            "code_regression",
            "corpus, configuration and probes are identical; the pipeline changed",
        )
        evidence.append("manifest_id and config_fingerprint both match")

    attribution = Attribution(
        verdict=verdict,
        summary=summary,
        evidence=tuple(evidence),
        corpus_changed=corpus_changed,
        config_changed=config_changed,
        dataset_changed=dataset_changed,
    )

    # -- per-probe regressions ---------------------------------------------- #
    regressions: list[ProbeRegression] = []
    improvements: list[str] = []

    by_id = probes.by_id if probes else None
    before_map = baseline.by_id
    after_map = candidate.by_id

    for probe_id, after_outcome in after_map.items():
        before_outcome = before_map.get(probe_id)
        if before_outcome is None:
            continue
        if before_outcome.passed and not after_outcome.passed:
            probe = by_id(probe_id) if by_id else None
            if probe is None:
                continue
            regressions.append(
                _attribute_probe(probe, before_outcome, after_outcome, diff, config_changed)
            )
        elif not before_outcome.passed and after_outcome.passed:
            improvements.append(probe_id)

    # -- metric deltas ------------------------------------------------------ #
    k = candidate.k or baseline.k
    b_agg, c_agg = baseline.aggregate, candidate.aggregate
    # A list, not a tuple: the leakage delta below is conditional, and a tuple
    # has no `append`. The parenthesised form silently produced a tuple, so
    # comparing two runs where the leak count differed -- the single most
    # important comparison this module exists to support -- raised
    # `AttributeError: 'tuple' object has no attribute 'append'` instead of
    # reporting the leak.
    metric_deltas: list[MetricDelta] = [
        MetricDelta("pass_rate", b_agg.pass_rate, c_agg.pass_rate),
        MetricDelta(f"recall_at_{k}", b_agg.recall_at_k, c_agg.recall_at_k),
        MetricDelta(f"ndcg_at_{k}", b_agg.ndcg_at_k, c_agg.ndcg_at_k),
        MetricDelta("mrr", b_agg.mrr, c_agg.mrr),
        MetricDelta("fact_coverage", b_agg.fact_coverage, c_agg.fact_coverage),
        MetricDelta("context_precision", b_agg.context_precision, c_agg.context_precision),
        MetricDelta("abstention_accuracy", b_agg.abstention_accuracy, c_agg.abstention_accuracy),
        # Latency and leakage are the two metrics where an *increase* is the
        # failure. They carry `higher_is_better=False` so an increase registers.
        # Latency keeps a 1ms tolerance for sub-millisecond measurement jitter;
        # leakage gets none -- there is no noise floor on "documents this caller
        # was not allowed to see".
        MetricDelta(
            "p95_latency_ms",
            b_agg.p95_latency_ms,
            c_agg.p95_latency_ms,
            tolerance=1.0,
            higher_is_better=False,
        ),
    ]
    if b_agg.leakage_count != c_agg.leakage_count:
        metric_deltas.append(
            MetricDelta(
                "leakage_count",
                float(b_agg.leakage_count),
                float(c_agg.leakage_count),
                higher_is_better=False,
            )
        )

    per_kind: dict[str, dict[str, float]] = {}
    for kind in sorted(set(b_agg.by_kind) | set(c_agg.by_kind)):
        b_kind = b_agg.by_kind.get(kind, {})
        c_kind = c_agg.by_kind.get(kind, {})
        entry = {
            "pass_rate_baseline": b_kind.get("pass_rate", 0.0),
            "pass_rate_candidate": c_kind.get("pass_rate", 0.0),
        }
        entry["pass_rate_delta"] = entry["pass_rate_candidate"] - entry["pass_rate_baseline"]
        per_kind[kind] = entry

    return RegressionReport(
        baseline=baseline,
        candidate=candidate,
        attribution=attribution,
        manifest_diff=diff,
        regressions=tuple(regressions),
        improvements=tuple(improvements),
        # A tuple because the report field is declared as one, and a report that
        # can be appended to after the fact is not the immutable record it is
        # supposed to be.
        metric_deltas=tuple(metric_deltas),
        per_kind_deltas=per_kind,
    )


def compare(
    baseline: RunReport | str | os.PathLike,
    candidate: RunReport | str | os.PathLike,
    *,
    probes: ProbeSet | str | os.PathLike | None = None,
    manifest_before: CorpusManifest | str | os.PathLike | None = None,
    manifest_after: CorpusManifest | str | os.PathLike | None = None,
) -> RegressionReport:
    """`attribute`, but also accepting paths.

    Comparing two runs is a command-line operation in practice, and threading
    four file loads through a shell is where a comparison goes wrong: it is very
    easy to diff the current run against a report generated against a different
    corpus and never notice. Passing the manifests alongside the reports is what
    keeps that pairing honest, so the file paths are accepted here rather than
    left to the caller to assemble.

    `os.PathLike` is accepted alongside `str` because the natural thing to pass
    from a caller that built the paths with `pathlib` is a `Path`, and rejecting
    it would push every user into a pointless `str()` conversion at the call
    site.
    """
    def _load(value: Any, loader: Callable[[Any], Any]) -> Any:
        if value is None:
            return None
        return loader(value) if isinstance(value, (str, os.PathLike)) else value

    base = _load(baseline, RunReport.load)
    cand = _load(candidate, RunReport.load)
    probe_set = _load(probes, ProbeSet.load)
    before_manifest = _load(manifest_before, CorpusManifest.load)
    after_manifest = _load(manifest_after, CorpusManifest.load)

    return attribute(
        base,
        cand,
        probes=probe_set,
        manifest_before=before_manifest,
        manifest_after=after_manifest,
    )
