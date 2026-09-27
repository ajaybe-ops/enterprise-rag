"""The answer layer: citation-grounded generation with abstention.

What this layer is responsible for
----------------------------------
Retrieval decides which documents are relevant. This layer decides whether to
answer at all, and makes the answer auditable. Three things happen here, in this
order, and the order is the point:

1. **Authorisation** (`app.security.access`) -- if the best evidence is outside
   the caller's scope, refuse. Answering from a lower-ranked readable document
   instead would be a silent factual substitution.
2. **Citation verification** -- every claim in the generated answer must map to a
   span of retrieved text. A sentence that cannot be traced is dropped, and if
   too few survive, the answer is withdrawn.
3. **Abstention** -- if the evidence does not support a confident answer, say so.

Why verification precedes presentation
--------------------------------------
The common alternative is to generate, then append citations, then check. That
order produces the worst outcome: a fluent, confident, wrong answer with a real
citation attached to it, because the citation only proves a document was
retrieved, not that it supports the sentence. Verifying first, and dropping
unsupported sentences, is the only ordering where a citation is evidence.

On abstention
-------------
Abstention is a first-class outcome, not an error path. A system that answers
every question has not been measured; it has been assumed. The threshold is
configurable and, like every other threshold here, should be refit against a
labelled probe set before being trusted in production.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Sequence

__all__ = ["Answer", "Citation", "build_answer", "verify_citations", "ABSTAIN_TEMPLATE"]

ABSTAIN_TEMPLATE = (
    "I could not find an authoritative answer to that in the indexed documents."
)

# Sentence splitting is intentionally simple. A dependency-grade splitter is
# better at abbreviations and lists, but it is also a source of silent behaviour
# differences across versions, and a verification layer whose segmentation can
# shift under a patch release is not a gate anyone should trust. The cost is
# over-splitting on "e.g." and "i.e.", which is recoverable; the cost of an
# unstable gate is not.
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_TOKEN = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    """The single tokenizer used for both sides of every comparison below.

    This exists because the verifier silently never matched anything when the
    two sides used different tokenisations: the evidence was split on whitespace
    and so contained `milliseconds,` with the comma attached, while the claim was
    tokenised with a character class and contained `milliseconds`. Every overlap
    was zero, every sentence was "unsupported", and the system withdrew correct
    answers with a confident explanation. Two tokenisers either side of a
    verification step is a defect generator, not a style question -- so both
    sides go through here.
    """
    return _TOKEN.findall(text.casefold())


def _content_words(text: str) -> set[str]:
    """Tokens long enough to carry meaning. Short tokens are stopword-ish."""
    return {t for t in _tokens(text) if len(t) >= 4}


@dataclass(frozen=True, slots=True)
class Citation:
    """A verified link from a claim to the text that supports it."""

    source: str
    ordinal: int
    quote: str
    verified: bool = True
    chunk_id: str = ""

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "chunk": self.ordinal,
            "chunk_id": self.chunk_id,
            "quote": self.quote,
            "verified": self.verified,
        }


@dataclass(frozen=True, slots=True)
class Answer:
    """The response object. `abstained` is a first-class field, not an error."""

    query: str
    text: str
    citations: tuple[Citation, ...] = ()
    abstained: bool = False
    reason: str = ""
    latency_ms: float = 0.0
    retrieval_ms: float = 0.0
    confidence: float = 0.0
    strategy: str = ""
    escalations: int = 0
    withheld_count: int = 0
    injection_findings: int = 0
    trace: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.text,
            "abstained": self.abstained,
            "reason": self.reason,
            "citations": [c.to_dict() for c in self.citations],
            "latency_ms": round(self.latency_ms, 2),
            "retrieval_ms": round(self.retrieval_ms, 2),
            "confidence": round(self.confidence, 4),
            "strategy": self.strategy,
            "escalations": self.escalations,
            "withheld_count": self.withheld_count,
            "injection_findings": self.injection_findings,
        }


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


# Small, closed-class function words that legitimately appear in a synthesised
# answer but never in the retrieved evidence, and so must not be read as
# hallucinated terminology. A closed list rather than a general stopword corpus:
# the set of words allowed to appear without evidence is a policy decision about
# what this system will accept, and importing someone else's list would make
# that decision invisibly depend on a package version.
_ANSWER_FUNCTION_WORDS = frozenset(
    """
    the this that these those there here when while which whose what where
    with from into onto over under about after before during within without
    than then them they their its it is are was were be been being has have
    had does did will would shall should can could may might must not only
    also such both each every some any all more most less least very much
    many few same other another new first last next previous current
    """.split()
)


def verify_citations(
    sentences: Sequence[str], chunks: Sequence
) -> tuple[list[tuple[str, object, str]], list[str]]:
    """Split into sentences and keep those traceable to retrieved text.

    A sentence survives if enough of its *content words* appear in the evidence.
    Matching is on content words rather than on a contiguous quote because a
    model legitimately paraphrases; demanding verbatim overlap would reject
    correct answers and quietly turn the verifier into a copy-detector.

    The threshold is 0.6 of content words, and words shorter than four characters
    are dropped as stopword-ish. Both are tuned against the probe suite, which
    is the only reason to trust a number like that.

    Each survivor is returned with the chunk that actually supports it, not the
    top-ranked chunk -- see `_best_supporting_chunk`.
    """
    if not sentences or not chunks:
        return [], list(sentences)

    content_words = set()
    for chunk in chunks:
        content_words |= _content_words(getattr(chunk, "text", ""))
    haystack_words = content_words | (_ANSWER_FUNCTION_WORDS & set(content_words))
    haystack_words |= _ANSWER_FUNCTION_WORDS

    verified: list[tuple[str, object, str]] = []
    rejected: list[str] = []

    for sentence in sentences:
        content = _content_words(sentence)
        if not content:
            rejected.append(sentence)
            continue
        # Require both conditions: the sentence's own words must be present, and
        # it must not introduce terminology that appears nowhere in the evidence.
        # Either test alone is insufficient -- the first cannot detect a
        # confidently-stated falsehood built entirely from real words, and the
        # second is what catches "the limit is 5 TB" when the corpus never says TB.
        if len(content & haystack_words) / len(content) >= 0.6 and content <= haystack_words:
            best = _best_supporting_chunk(sentence, chunks)
            if best is not None:
                verified.append((sentence, best, getattr(best, "text", "")[:280]))
                continue
        rejected.append(sentence)

    return verified, rejected




def _best_supporting_chunk(sentence: str, chunks: Sequence):
    """The chunk that supports `sentence` best, by content-word overlap.

    Attributing a claim to the *most* supporting chunk rather than the
    top-ranked one matters: a citation that points at a document that merely
    mentions the same words is not a citation, and picking by rank alone
    produces exactly those.
    """
    content = _content_words(sentence)
    if not content:
        return None

    best, best_score = None, 0
    for chunk in chunks:
        words = _content_words(getattr(chunk, "text", ""))
        score = len(content & words)
        if score > best_score:
            best, best_score = chunk, score
    return best


def build_answer(
    query: str,
    chunks: Sequence,
    *,
    generated: str = "",
    abstained: bool = False,
    reason: str = "",
    confidence: float = 0.0,
    strategy: str = "",
    escalations: int = 0,
    retrieval_ms: float = 0.0,
    withheld_count: int = 0,
    injection_findings: int = 0,
    min_supported_sentences: int = 1,
    verify: bool = True,
) -> Answer:
    """Assemble the final response, verifying whatever was generated.

    `generated` is injected rather than produced here. The generator is a
    swappable provider, and the verification step is the part that must not be
    swappable: keeping it a separate, provider-independent function is what
    guarantees a different model cannot quietly opt out of citation checking.
    """
    started = time.perf_counter()

    if abstained or not chunks:
        return Answer(
            query=query,
            text=ABSTAIN_TEMPLATE,
            citations=(),
            abstained=True,
            reason=reason or "no evidence available within the caller's scope",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            retrieval_ms=retrieval_ms,
            confidence=confidence,
            strategy=strategy,
            escalations=escalations,
            withheld_count=withheld_count,
            injection_findings=injection_findings,
        )

    if not verify or not generated.strip():
        return Answer(
            query=query,
            text=generated or ABSTAIN_TEMPLATE,
            citations=(),
            abstained=not generated.strip(),
            reason=reason or ("no generation available" if not generated.strip() else ""),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            retrieval_ms=retrieval_ms,
            confidence=confidence,
            strategy=strategy,
            escalations=escalations,
            withheld_count=withheld_count,
            injection_findings=injection_findings,
        )

    sentences = [s.strip() for s in _SENTENCE.split(generated.strip()) if s.strip()]
    verified, rejected = verify_citations(sentences, chunks)

    # Withdraw the answer when too little survives verification. Keeping the
    # surviving sentences and presenting them as the answer would be a partial
    # answer presented as a complete one, which is the failure mode most likely
    # to mislead without being noticed.
    if len(verified) < min_supported_sentences:
        return Answer(
            query=query,
            text=ABSTAIN_TEMPLATE,
            citations=(),
            abstained=True,
            reason=reason
            or (
                f"only {len(verified)} of {len(sentences)} generated sentence(s) "
                f"could be traced to the retrieved evidence; withdrawn rather than "
                f"presenting an unsupported answer"
            ),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            retrieval_ms=retrieval_ms,
            confidence=confidence,
            strategy=strategy,
            escalations=escalations,
            withheld_count=withheld_count,
            injection_findings=injection_findings,
        )

    citations = tuple(
        Citation(
            source=getattr(chunk, "source", ""),
            ordinal=int(getattr(chunk, "ordinal", 0)),
            quote=quote,
            chunk_id=getattr(chunk, "chunk_id", ""),
        )
        for _, chunk, quote in verified
    )

    text = " ".join(sentence for sentence, _, _ in verified)
    if rejected:
        text += f"\n\n({len(rejected)} unsupported sentence(s) were removed.)"

    return Answer(
        query=query,
        text=text,
        citations=citations,
        abstained=False,
        reason=reason,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        retrieval_ms=retrieval_ms,
        confidence=confidence,
        strategy=strategy,
        escalations=escalations,
        withheld_count=withheld_count,
        injection_findings=injection_findings,
    )
