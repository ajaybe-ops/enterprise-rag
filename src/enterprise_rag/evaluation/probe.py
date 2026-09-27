"""Probes: the unit of retrieval quality.

Why probes and not a relevance judgement
---------------------------------------
The standard IR evaluation asks "is this document relevant to this query?" and
scores a ranked list. That is the right primitive and this project uses it, but
on its own it cannot express the failures that actually page an on-call engineer
in a RAG deployment:

* The right *document* was retrieved, but the specific *passage* carrying the
  fact was not, so the model answered from the wrong paragraph of the right
  file. Relevance judgement scores this as a success.
* A *superseded* revision of a document outranked the current one. Every
  relevance label is satisfied -- the topic matches perfectly -- and the answer
  is confidently wrong about a number that changed last quarter.
* A document the caller is not authorised to read surfaced in the context. No
  relevance metric penalises this, because relevance is not a security property.
* The query is genuinely unanswerable and the system should have abstained. Any
  ranked list is "correct" here.

So a probe declares a *contract* with five independent obligations, each of which
can fail on its own:

    must_retrieve   documents whose absence makes the answer ungroundable
    must_contain    literal facts that must appear in the assembled context
    must_not        documents that must not appear at all (stale, denied)
    must_outrank    documents that must rank above the `must_not` set
    should_abstain  whether the correct behaviour is to refuse

`must_contain` is deliberately literal-substring matching rather than semantic
similarity. It is not a proxy for "the model understood the passage" -- it is a
check that the pipeline *selected the right span*, which is fully deterministic
and therefore usable as a CI gate. Anything requiring judgement belongs in a
separate, explicitly non-gating suite; see docs/adr/0003.

Why `must_outrank` is separate from `must_not`
----------------------------------------------
`must_not` means *absence*, and absence is the right contract for exactly one
thing: a document the caller must never see, where any appearance is a
disclosure.

It is the wrong contract for competing documents. Writing "the draft must not be
returned alongside the SLO" to express "the draft must win" makes a correct
system fail for returning a second relevant document -- and the suite then
rewards a retriever that is confidently narrow. The evaluation suite found this
in its own first run: a lexical-trap probe was marked failing while the draft sat
at rank 1 and the distractor at rank 2, because the contract demanded the
distractor be absent rather than outranked.

So intent discrimination is expressed as precedence, and absolute exclusion is
reserved for authorisation. The two are different properties and collapsing them
produces a suite that is either too strict to pass or too loose to catch
anything.

Reproducibility
---------------
A probe set is a versioned JSON document. `ProbeSet.validate()` refuses to load a
set that is internally inconsistent (a probe demanding a document that is also
forbidden, an abstention probe with required evidence) because a contradictory
contract produces a metric that looks rigorous and is meaningless.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "PROBE_KINDS",
    "Probe",
    "ProbeKind",
    "ProbeOutcome",
    "ProbeSet",
    "load_probe_set",
]

# Each kind names a *failure mode being probed*, not a question category. That
# distinction matters when reading a report: "kinds: 4 version_conflict, 3
# lexical_trap" says something actionable about corpus risk, whereas "kinds:
# 7 technical, 5 general" does not.
ProbeKind = Literal[
    "direct",           # answerable, single hop, plainly worded
    "paraphrase",       # answerable, no shared vocabulary with the source
    "multi_hop",        # answerable, needs two documents to be conclusive
    "version_conflict", # a superseded revision is in the corpus and must lose
    "lexical_trap",     # a high-term-overlap distractor must not win
    "unanswerable",     # nothing in the corpus supports an answer
    "forbidden",        # the answer exists but the caller may not see it
]

PROBE_KINDS: tuple[str, ...] = (
    "direct",
    "paraphrase",
    "multi_hop",
    "version_conflict",
    "lexical_trap",
    "unanswerable",
    "forbidden",
)


@dataclass(frozen=True, slots=True)
class Probe:
    """One query plus the obligations its answer must satisfy."""

    probe_id: str
    query: str
    kind: ProbeKind = "direct"

    # The retrieval contract.
    must_retrieve: tuple[str, ...] = ()
    must_contain: tuple[str, ...] = ()
    must_not: tuple[str, ...] = ()
    must_outrank: tuple[str, ...] = ()

    # The abstention contract. Implied by kind == "unanswerable" but
    # explicit so a probe set can be written by hand without magic.
    should_abstain: bool = False
    # The authorisation context. A probe with a tenant that differs from the
    # serving tenant is an access-control test, and `forbidden` kind is the
    # canonical case.
    tenant: str = "public"
    subject: str = "public"

    # Whether `must_not` encodes an access-control rule or a relevance
    # preference. These are not the same severity and must not be reported in one
    # number: conflating them means a run with 24 superseded-document
    # preference violations and 0 unauthorised disclosures reads identically to
    # the reverse, and the one that matters gets averaged away.
    must_not_kind: Literal["authorization", "preference"] = "preference"

    # Free-text provenance: why this probe exists, what bug it was written for.
    # Carried into reports so a failing probe explains itself without a lookup.
    rationale: str = ""
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in PROBE_KINDS:
            raise ValueError(f"probe {self.probe_id}: unknown kind {self.kind!r}")
        if self.should_abstain and (self.must_retrieve or self.must_contain):
            raise ValueError(
                f"probe {self.probe_id}: declares should_abstain but also requires "
                "evidence. An unanswerable query cannot have must_retrieve, and a "
                "contract that demands both is measuring nothing."
            )
        overlap = set(self.must_retrieve) & set(self.must_not)
        if overlap:
            raise ValueError(
                f"probe {self.probe_id}: {sorted(overlap)} appear in both "
                "must_retrieve and must_not; the contract is unsatisfiable."
            )
        if self.must_outrank and not self.must_not:
            raise ValueError(
                f"probe {self.probe_id}: must_outrank is meaningless without a "
                "must_not set to outrank. A precedence obligation needs something "
                "to take precedence over."
            )
        precedence_clash = set(self.must_outrank) & set(self.must_not)
        if precedence_clash:
            raise ValueError(
                f"probe {self.probe_id}: {sorted(precedence_clash)} must both "
                "outrank and be excluded."
            )
        # Default the severity from the kind when it was not stated, so a
        # hand-written probe set cannot silently classify an access-control
        # expectation as a mere preference and have it reported as a warning.
        if (
            self.must_not
            and self.kind in {"forbidden", "unanswerable"}
            and self.must_not_kind == "preference"
        ):
            object.__setattr__(self, "must_not_kind", "authorization")

    @property
    def is_negative(self) -> bool:
        """Probes where the *correct* outcome is to surface nothing or refuse."""
        return self.should_abstain or self.kind in {"unanswerable", "forbidden"}

    def to_dict(self) -> dict:
        return {
            "probe_id": self.probe_id,
            "query": self.query,
            "kind": self.kind,
            "must_retrieve": list(self.must_retrieve),
            "must_contain": list(self.must_contain),
            "must_not": list(self.must_not),
            "must_outrank": list(self.must_outrank),
            "must_not_kind": self.must_not_kind,
            "should_abstain": self.should_abstain,
            "tenant": self.tenant,
            "subject": self.subject,
            "rationale": self.rationale,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> Probe:
        return cls(
            probe_id=payload["probe_id"],
            query=payload["query"],
            kind=payload.get("kind", "direct"),
            must_retrieve=tuple(payload.get("must_retrieve", ())),
            must_contain=tuple(payload.get("must_contain", ())),
            must_not=tuple(payload.get("must_not", ())),
            must_outrank=tuple(payload.get("must_outrank", ())),
            must_not_kind=payload.get("must_not_kind", "preference"),
            should_abstain=bool(payload.get("should_abstain", False)),
            tenant=payload.get("tenant", "public"),
            subject=payload.get("subject", "public"),
            rationale=payload.get("rationale", ""),
            tags=tuple(payload.get("tags", ())),
        )


@dataclass(frozen=True, slots=True)
class ProbeSet:
    """A versioned, validated collection of probes.

    `dataset_id` hashes the probe contents. Recording it next to a result is what
    makes a number reproducible: "recall@5 = 0.91 on dataset 3f2a..." is a claim
    that can be re-checked, whereas "recall@5 = 0.91" is a rumour.
    """

    probes: tuple[Probe, ...]
    name: str = "unnamed"
    dataset_id: str = ""
    corpus_manifest_id: str = ""
    notes: str = ""

    def __len__(self) -> int:
        return len(self.probes)

    def __iter__(self):
        return iter(self.probes)

    def by_id(self, probe_id: str) -> Probe | None:
        for probe in self.probes:
            if probe.probe_id == probe_id:
                return probe
        return None

    def of_kind(self, kind: str) -> tuple[Probe, ...]:
        return tuple(p for p in self.probes if p.kind == kind)

    @property
    def positive(self) -> tuple[Probe, ...]:
        return tuple(p for p in self.probes if not p.is_negative)

    @property
    def negative(self) -> tuple[Probe, ...]:
        return tuple(p for p in self.probes if p.is_negative)

    @property
    def kind_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for probe in self.probes:
            counts[probe.kind] = counts.get(probe.kind, 0) + 1
        return dict(sorted(counts.items()))

    def validate(self, known_sources: Iterable[str] | None = None) -> list[str]:
        """Return a list of contract errors. Empty means the set is coherent.

        Accepts a set of known sources so a typo in a document name is caught at
        load time rather than as a permanent, silent 0% recall on one probe --
        which is otherwise very easy to mistake for a retrieval bug.
        """
        problems: list[str] = []
        sources = set(known_sources) if known_sources is not None else None
        seen: set[str] = set()

        for probe in self.probes:
            if probe.probe_id in seen:
                problems.append(f"duplicate probe_id {probe.probe_id!r}")
            seen.add(probe.probe_id)
            if not probe.query.strip():
                problems.append(f"probe {probe.probe_id}: empty query")
            if sources is None:
                continue
            for group, label in (
                (probe.must_retrieve, "must_retrieve"),
                (probe.must_not, "must_not"),
            ):
                for ref in group:
                    if ref not in sources:
                        problems.append(
                            f"probe {probe.probe_id}: {label} references unknown "
                            f"source {ref!r}"
                        )
        return problems

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dataset_id": self.dataset_id,
            "corpus_manifest_id": self.corpus_manifest_id,
            "notes": self.notes,
            "probe_count": len(self.probes),
            "kind_counts": self.kind_counts,
            "probes": [p.to_dict() for p in self.probes],
        }

    def save(self, path: str | pathlib.Path) -> pathlib.Path:
        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["dataset_id"] = dataset_id_for(payload["probes"])
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return target

    @classmethod
    def load(cls, path: str | pathlib.Path, *, validate_against: Iterable[str] | None = None) -> ProbeSet:
        path = pathlib.Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        probes = tuple(Probe.from_dict(p) for p in payload.get("probes", ()))
        result = cls(
            probes=probes,
            name=payload.get("name", path.stem),
            dataset_id=payload.get("dataset_id") or dataset_id_for(payload.get("probes", [])),
            corpus_manifest_id=payload.get("corpus_manifest_id", ""),
            notes=payload.get("notes", ""),
        )
        if validate_against is not None:
            problems = result.validate(validate_against)
            if problems:
                raise ValueError(
                    f"{path}: {len(problems)} contract problem(s):\n  "
                    + "\n  ".join(problems)
                )
        return result


@dataclass(frozen=True, slots=True)
class ProbeOutcome:
    """The measured result of running one probe.

    Every obligation is recorded as its own boolean rather than a single
    "passed" flag. A single flag is what makes regression reports useless: it
    cannot distinguish "the right document arrived but the fact did not" from
    "nothing arrived at all", and those two have different causes.
    """

    probe_id: str
    kind: str
    query: str

    retrieved_sources: tuple[str, ...] = ()
    context_text: str = ""

    # Obligation results.
    recall_hit: bool = False
    facts_hit: bool = False
    clean: bool = True          # no forbidden source surfaced
    precedence_ok: bool = True  # must_outrank docs all precede must_not docs
    abstained: bool = False
    passed: bool = False

    # The two must_not outcomes, kept apart. `authz_violation` is a security
    # event; `preference_violation` is a ranking quality problem. A run that
    # reports one number for both makes the second look like the first.
    authz_violation: bool = False
    preference_violation: bool = False

    # Ranking detail, for attribution.
    first_required_rank: int | None = None
    missing_sources: tuple[str, ...] = ()
    missing_facts: tuple[str, ...] = ()
    leaked_sources: tuple[str, ...] = ()

    # Cost.
    latency_ms: float = 0.0
    candidates_considered: int = 0
    confidence: float = 0.0
    escalations: int = 0
    strategy: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "probe_id": self.probe_id,
            "kind": self.kind,
            "query": self.query,
            "retrieved_sources": list(self.retrieved_sources),
            "recall_hit": self.recall_hit,
            "facts_hit": self.facts_hit,
            "clean": self.clean,
            "precedence_ok": self.precedence_ok,
            "authz_violation": self.authz_violation,
            "preference_violation": self.preference_violation,
            "abstained": self.abstained,
            "passed": self.passed,
            "first_required_rank": self.first_required_rank,
            "missing_sources": list(self.missing_sources),
            "missing_facts": list(self.missing_facts),
            "leaked_sources": list(self.leaked_sources),
            "latency_ms": round(self.latency_ms, 2),
            "candidates_considered": self.candidates_considered,
            "confidence": round(self.confidence, 4),
            "escalations": self.escalations,
            "strategy": self.strategy,
            "error": self.error,
        }


def dataset_id_for(probe_payloads: Sequence[Mapping]) -> str:
    """Stable id over probe content, excluding free-text rationale.

    Rationale and tags are documentation. Including them would change the id every
    time someone reworded a comment, breaking the link between a result and the
    dataset that produced it.
    """
    material = json.dumps(
        [
            {
                "probe_id": p.get("probe_id"),
                "query": p.get("query"),
                "kind": p.get("kind"),
                "must_retrieve": sorted(p.get("must_retrieve", ())),
                "must_contain": sorted(p.get("must_contain", ())),
                "must_not": sorted(p.get("must_not", ())),
                "must_outrank": sorted(p.get("must_outrank", ())),
                "must_not_kind": p.get("must_not_kind", "preference"),
                "should_abstain": bool(p.get("should_abstain", False)),
                "tenant": p.get("tenant", "public"),
            }
            for p in probe_payloads
        ],
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


def load_probe_set(
    path: str | pathlib.Path, *, validate_against: Iterable[str] | None = None
) -> ProbeSet:
    """Load and validate a probe set. Raises on an incoherent contract.

    Refusing to load is deliberate. A probe set that cannot be satisfied trains
    the reader to ignore the gate, and a gate that is always red is worse than no
    gate because it looks like the system is broken rather than the dataset.
    """
    return ProbeSet.load(path, validate_against=validate_against)
