"""Citation data types.

A citation here is not a URL appended to a sentence. It is a *verified* link
from one claim to the span of retrieved text that supports it, which is why
`verified` exists as a field at all: the honest state includes "claimed but not
confirmed", and collapsing that to a boolean at the presentation layer is how
systems end up rendering unchecked citations with the same styling as checked
ones.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Citation", "CitationReport"]


@dataclass(frozen=True, slots=True)
class Citation:
    """A verified link from a claim to the text that supports it."""

    source: str
    ordinal: int
    quote: str
    chunk_id: str = ""
    verified: bool = True

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "chunk": self.ordinal,
            "chunk_id": self.chunk_id,
            "quote": self.quote,
            "verified": self.verified,
        }


@dataclass(frozen=True, slots=True)
class CitationReport:
    """The outcome of checking generated text against retrieved evidence.

    `coverage` is the fraction of generated sentences that survived. It is
    reported separately from the answer because the two answer different
    questions: a low-coverage answer that was nonetheless served is a system
    defect, while a high-coverage abstention is the system working. An operator
    watching only `abstained` cannot tell those apart.
    """

    citations: tuple[Citation, ...] = ()
    verified_sentences: tuple[str, ...] = ()
    rejected_sentences: tuple[str, ...] = ()
    total_sentences: int = 0

    @property
    def coverage(self) -> float:
        if self.total_sentences == 0:
            return 0.0
        return len(self.verified_sentences) / self.total_sentences

    @property
    def supported(self) -> bool:
        return bool(self.verified_sentences)

    def to_dict(self) -> dict:
        return {
            "citations": [c.to_dict() for c in self.citations],
            "verified_sentences": list(self.verified_sentences),
            "rejected_sentences": list(self.rejected_sentences),
            "total_sentences": self.total_sentences,
            "coverage": round(self.coverage, 4),
            "supported": self.supported,
        }
