"""Document-status awareness.

The problem
-----------
A corpus that has been edited over a few years contains documents that are
explicitly wrong for present-tense questions: superseded objectives, retired
runbooks, draft RFCs. They are usually *lexically excellent* matches -- often
better than the current document, because a document that has been rewritten
three times accumulates every phrasing the team has ever used for the topic.

A retriever with no notion of document status will therefore prefer the wrong
document, confidently, and cite it. This is not hypothetical and it is not a
rare edge case: it is the default behaviour of any system that ranks on text
similarity alone, and it is why "the AI told me the old number" is a common
complaint about deployed RAG.

The policy
----------
Two different treatments, because the two cases are genuinely different:

    superseded  excluded from the candidate pool
    draft       demoted by a score multiplier

`superseded` is excluded rather than demoted because the document says so
itself: it is retained for audit history, and its content is known to be wrong.
Ranking a known-wrong document highly because it ranks lexically well is
precisely the failure. A demotion large enough to usually beat it is a
demotion that occasionally does not, and the occasions are the incidents.

`draft` is only demoted, never excluded, because drafts are sometimes the only
source of truth -- an unreleased feature is documented in the draft and nowhere
else. Excluding drafts would make the system confidently say a feature does not
exist.

Both lists are configuration. Which statuses count as superseded is a property
of a given document collection, not a law, and hard-coding a vocabulary the
operator cannot extend guarantees the policy is wrong at their site.

Query-conditioned demotion
--------------------------
A global demotion multiplier is wrong, and the evaluation suite is what proved
it. Demoting drafts correctly fixes the case where a draft competes with a
current document ("what is the p99 objective?"), and correctly *breaks* the
case where the draft is the answer ("what p99 latency has been proposed?"). One
constant cannot be right for both, because the authority of a document is not a
property of the document -- it is a property of the document *relative to the
question*.

So the multiplier is conditioned on the query. When the query is itself asking
about preliminary material, the demotion is suspended. This is not a fudge to
make a test pass; it is the observation that "is this the authoritative source"
is a question about the pair (document, query), and any policy that only looks
at the document half of that pair is going to be wrong about some queries.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

__all__ = ["DEFAULT_DEMOTED", "DEFAULT_SUPERSEDED", "PRELIMINARY_INTENT", "StatusPolicy"]

# Statuses whose content is known to be wrong for present-tense questions.
DEFAULT_SUPERSEDED: tuple[str, ...] = ("superseded", "retired", "archived", "deprecated")

# Statuses that are worth demoting but must stay reachable.
DEFAULT_DEMOTED: tuple[str, ...] = ("draft", "proposed", "experimental", "unapproved")

# Terms that indicate the *asker* wants preliminary material. When one of these
# is present, demotion is suspended: the question is explicitly about a proposal
# or a draft, so the draft is the authority for it and the current document is
# the wrong answer.
PRELIMINARY_INTENT = re.compile(
    r"\b(?:proposed|proposal|propose[sd]?|draft|drafting|rfc|"
    r"under\s+consideration|being\s+considered|initial\s+idea|"
    r"not\s+(?:yet\s+)?(?:approved|adopted|ratified)|"
    r"what\s+has\s+been\s+tentatively)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class StatusPolicy:
    """Applies exclusion and demotion from document `status` metadata.

    A document with no status is neither excluded nor demoted. Defaulting an
    unknown status to "draft" would quietly demote most of a corpus that simply
    does not use front matter, and defaulting it to "superseded" would delete
    the corpus. Absent metadata has to mean "unclassified", and the operator
    decides what to do with that.
    """

    superseded: frozenset[str] = frozenset(DEFAULT_SUPERSEDED)
    demoted: frozenset[str] = frozenset(DEFAULT_DEMOTED)
    demotion: float = 0.55

    def __post_init__(self) -> None:
        # Overlap would make the two rules contradictory and the outcome would
        # depend on evaluation order. Refusing the config is better than picking
        # silently.
        overlap = self.superseded & self.demoted
        if overlap:
            raise ValueError(
                f"statuses {sorted(overlap)} are both excluded and demoted; the "
                "policy would be ambiguous"
            )
        if not 0.0 < self.demotion <= 1.0:
            raise ValueError(f"demotion must be in (0, 1], got {self.demotion}")

    @classmethod
    def from_settings(cls, settings: object) -> StatusPolicy:
        retrieval = getattr(settings, "retrieval", None)
        return cls(
            superseded=frozenset(
                getattr(retrieval, "status_exclude", None) or DEFAULT_SUPERSEDED
            ),
            demoted=frozenset(getattr(retrieval, "status_demote", None) or DEFAULT_DEMOTED),
            demotion=float(getattr(retrieval, "status_demotion", 0.55) or 0.55),
        )

    def is_superseded(self, status: str | None) -> bool:
        return (status or "").strip().lower() in self.superseded

    def is_demoted(self, status: str | None) -> bool:
        return (status or "").strip().lower() in self.demoted

    def is_excluded(self, metadata: dict | None) -> bool:
        return self.is_superseded((metadata or {}).get("status"))

    def multiplier(self, metadata: dict | None, query: str = "") -> float:
        """Score multiplier for a candidate. 1.0 means untouched.

        Applied to the fused score rather than to the lexical score, so a
        demoted document can still win when it is the only relevant one -- which
        is the point of demoting rather than excluding.

        `query` suspends the demotion when the question is explicitly about
        preliminary material. See the module docstring: authority is a property
        of the (document, query) pair, and a demotion that ignores the query
        half of it is wrong for every question of the form "what has been
        proposed?".
        """
        if not self.is_demoted((metadata or {}).get("status")):
            return 1.0
        if query and PRELIMINARY_INTENT.search(query):
            return 1.0
        return self.demotion

    def wants_preliminary(self, query: str) -> bool:
        return bool(query and PRELIMINARY_INTENT.search(query))

    def partition(self, chunks: Iterable) -> tuple[list, list, list]:
        """Split chunks into (kept, demoted, excluded), preserving order."""
        kept, demoted, excluded = [], [], []
        for chunk in chunks:
            raw = getattr(chunk, "metadata", None)
            metadata: dict[str, Any] = raw if isinstance(raw, dict) else {}
            # `is_excluded` takes the whole mapping; `is_demoted` takes the status
            # string. They are not interchangeable. Passing the mapping to
            # `is_demoted` -- which is what this did -- reaches
            # `(status or "").strip()` with a dict and raises
            # `AttributeError: 'dict' object has no attribute 'strip'` on the
            # first chunk that is not superseded. Nothing caught it because
            # `partition` has no caller, so the one line that used it was never
            # executed by a test or by the pipeline.
            status_value = metadata.get("status")
            status = status_value if isinstance(status_value, str) else None
            if self.is_excluded(metadata):
                excluded.append(chunk)
            elif self.is_demoted(status):
                demoted.append(chunk)
            else:
                kept.append(chunk)
        return kept, demoted, excluded

    def describe(self) -> str:
        return (
            f"exclude={sorted(self.superseded)} "
            f"demote={sorted(self.demoted)}x{self.demotion:g}"
        )
