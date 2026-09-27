"""Verification: does the evidence actually support the sentence?

Why this is a separate stage, and why it runs *before* the answer is presented
--------------------------------------------------------------------------------
The common alternative is generate, then append citations, then check. That order
produces the worst outcome available: a fluent, confident, wrong answer with a
real citation attached, because the citation only proves a document was
retrieved, not that it supports the claim. Verifying first and dropping
unsupported sentences is the only ordering in which a citation is evidence.

What "supported" means here
---------------------------
A sentence survives if enough of its *content words* appear in the evidence, and
it introduces no terminology found nowhere in the evidence. Both conditions are
required, and neither alone is sufficient:

* Overlap alone cannot detect a confidently-stated falsehood assembled entirely
  from real words ("retention is 30 days" when the corpus says 7).
* The novel-terminology test alone cannot detect a restatement that swaps a
  number for another real number from a different document.

Matching is on content words rather than a contiguous quote, because a model
legitimately paraphrases and demanding verbatim overlap turns the verifier into a
copy-detector that rejects correct answers. The price of that choice is that a
sentence reusing the corpus's vocabulary while asserting the wrong relation to it
can pass. The cost model, the alternative that catches it, and why it is not the
default are in `docs/evaluation.md`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from enterprise_rag.citations.models import Citation, CitationReport
from enterprise_rag.types import Chunk

__all__ = ["SUPPORT_THRESHOLD", "content_words", "split_sentences", "supporting_chunk", "verify"]

# Fraction of a sentence's content words that must appear in the evidence.
# Tuned against the probe suite, which is the only reason to trust a number like
# this; refit it when the corpus domain changes.
SUPPORT_THRESHOLD = 0.6

# Sentence splitting is intentionally simple. A dependency-grade splitter handles
# abbreviations and lists better, but it is also a source of silent behaviour
# drift across versions, and a verification stage whose segmentation can shift
# under a patch release is not a gate anyone should trust. The cost is
# over-splitting on "e.g." and "i.e.", which is recoverable.
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_TOKEN = re.compile(r"[a-z0-9']+")

# Block boundaries, applied before sentence splitting.
#
# The corpus is markdown, not prose, and the documents are hard-wrapped at ~80
# columns. Splitting the raw text on `[.!?]` alone therefore produced sentences
# several paragraphs long: the text between one full stop and the next
# contained blank lines, an ATX heading and two more sentences, because wrapped
# lines end in neither a full stop nor a capital. Two consequences, both bad.
#
#   * Generation quoted fragments containing "## Latency" mid-string.
#   * Verification compared one answer sentence against a 400-character blob, so
#     almost any claim scored as supported. Coverage numbers from that were
#     measuring the length of the evidence, not its support for the claim.
#
# Splitting on blank lines and heading boundaries first makes each unit a real
# sentence or a structural line, which `is_structural` can then classify.
_BLOCK_SPLIT = re.compile(r"\n[ \t]*\n|^(?=#{1,6}[ \t])", re.MULTILINE)

# Small, closed-class function words that legitimately appear in a synthesised
# answer but never in the retrieved evidence, and so must not be read as
# hallucinated terminology. A closed list rather than an imported stopword
# corpus: the set of words allowed to appear without evidence is a policy
# decision about what this system will accept, and importing someone else's list
# makes that decision depend invisibly on a package version.
_ALLOWED_FUNCTION_WORDS = frozenset(
    """
    the this that these those there here when while which whose what where
    with from into onto over under about after before during within without
    than then them they their its it is are was were be been being has have
    had does did will would shall should can could may might must not only
    also such both each every some any all more most less least very much
    many few same other another new first last next previous current
    """.split()
)


def split_sentences(text: str) -> list[str]:
    """Sentence units for `text`, markdown-aware.

    Blocks are separated on blank lines and heading boundaries before sentence
    punctuation is considered, so hard-wrapped prose yields real sentences and
    a heading is its own unit rather than the middle of a 400-character blob.
    Wrapped lines are re-joined with a single space: a newline inside a sentence
    is a layout artefact, and leaving it in makes a word broken across lines
    compare unequal to the same word in an answer.
    """
    units: list[str] = []
    for block in _BLOCK_SPLIT.split(text):
        if not block:
            continue
        collapsed = " ".join(line.strip() for line in block.strip().splitlines())
        collapsed = re.sub(r"[ \t]{2,}", " ", collapsed).strip()
        if not collapsed:
            continue
        units.extend(part.strip() for part in _SENTENCE.split(collapsed) if part.strip())
    return units


def content_words(text: str) -> set[str]:
    """Tokens long enough to carry meaning. Short tokens are stopword-ish."""
    return {t for t in _TOKEN.findall(text.casefold()) if len(t) >= 4}


# Markdown and YAML furniture that carries no claim, so must never be quoted as
# an answer nor counted as evidence for one.
_STRUCTURAL_PREFIXES = ("#", "---", "```", "|", ">", "- ", "* ", "[", "![")
_FRONT_MATTER_KEY = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(\S?)")


def is_structural(unit: str) -> bool:
    """True for a unit that is document furniture rather than a claim.

    Lives here, next to the splitter, because it is a property of the same
    segmentation decision: anything the splitter emits as its own block is a
    candidate, and the two questions ("what is a unit?", "which units are
    furniture?") only stay consistent if they are answered together.
    """
    stripped = unit.strip()
    if not stripped:
        return True
    if stripped.startswith(_STRUCTURAL_PREFIXES):
        return True
    # A bare `key: value` front-matter line, which has no prose after the colon.
    # A real sentence containing a colon ("The tradeoff: latency against recall")
    # has several words after it, and a longer key than these.
    match = _FRONT_MATTER_KEY.match(stripped)
    if match and len(match.group(1)) <= 24 and not match.group(2).strip():
        return True
    return False


def supporting_chunk(sentence: str, chunks: Sequence[Chunk]) -> Chunk | None:
    """The chunk that supports `sentence` best, by content-word overlap.

    Attributing a claim to the *most* supporting chunk rather than the top-ranked
    one matters: a citation pointing at a document that merely shares vocabulary
    is not a citation, and choosing by rank alone produces exactly those.
    """
    content = content_words(sentence)
    if not content:
        return None
    best: Chunk | None = None
    best_score = 0
    for chunk in chunks:
        score = len(content & content_words(chunk.text))
        if score > best_score:
            best, best_score = chunk, score
    return best


def verify(
    generated: str,
    chunks: Sequence[Chunk],
    *,
    threshold: float = SUPPORT_THRESHOLD,
    quote_chars: int = 280,
) -> CitationReport:
    """Check `generated` against `chunks`. Never raises; always reports.

    An empty `chunks` yields a report with everything rejected rather than an
    exception, so the caller's abstention path is data-driven instead of being
    split between an `if not chunks` branch here and a `try` there.
    """
    sentences = split_sentences(generated)
    if not sentences:
        return CitationReport(total_sentences=0)

    evidence_words: set[str] = set()
    for chunk in chunks:
        evidence_words |= content_words(chunk.text)
    # Function words the answer is allowed to introduce without evidence support.
    haystack = evidence_words | _ALLOWED_FUNCTION_WORDS

    citations: list[Citation] = []
    verified: list[str] = []
    rejected: list[str] = []

    for sentence in sentences:
        content = content_words(sentence)
        if not content:
            rejected.append(sentence)
            continue
        overlap = len(content & haystack) / len(content)
        introduces_new_terms = bool(content - evidence_words)
        if overlap >= threshold and not introduces_new_terms:
            chunk = supporting_chunk(sentence, chunks)
            if chunk is not None:
                citations.append(
                    Citation(
                        source=chunk.source,
                        ordinal=int(chunk.ordinal),
                        quote=chunk.text[:quote_chars],
                        chunk_id=chunk.chunk_id,
                    )
                )
                verified.append(sentence)
                continue
        rejected.append(sentence)

    return CitationReport(
        citations=tuple(citations),
        verified_sentences=tuple(verified),
        rejected_sentences=tuple(rejected),
        total_sentences=len(sentences),
    )
