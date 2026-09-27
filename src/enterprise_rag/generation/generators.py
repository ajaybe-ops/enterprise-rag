"""Answer generators: two implementations behind one interface.

Why there are exactly two
-------------------------
`ExtractiveGenerator` composes an answer from the retrieved sentences
themselves. It needs no credentials, is deterministic, and cannot introduce a
claim that is absent from the evidence -- which makes it the *default*, and the
baseline every other generator has to beat.

`ProviderGenerator` calls an OpenAI-compatible endpoint and is used when one is
configured. It can synthesise across documents, which the extractive path cannot
do, and it can also invent a claim no document contains, which is why everything
it returns goes through `citations.verifier` before it reaches a caller.

Why a protocol rather than an `if` statement
--------------------------------------------
The verification stage must not be something a generator can skip. With a
generator interface, `RagPipeline` holds the verifier and the generator as
separate collaborators, so swapping `ExtractiveGenerator` for
`ProviderGenerator` cannot change whether answers are checked -- only what is
checked. An `if use_llm:` branch inside one function is exactly the shape in
which that coupling creeps back in.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from enterprise_rag.citations.verifier import content_words, is_structural, split_sentences
from enterprise_rag.config.settings import Settings
from enterprise_rag.context.builder import ContextBlock, format_messages
from enterprise_rag.generation import provider
from enterprise_rag.generation.provider import LLMUnavailable
from enterprise_rag.types import Chunk, Usage

__all__ = [
    "ABSTAIN_TEMPLATE",
    "ExtractiveGenerator",
    "GenerationResult",
    "Generator",
    "ProviderGenerator",
    "select_generator",
]

ABSTAIN_TEMPLATE = "I could not find an authoritative answer to that in the indexed documents."

# How many retrieved sentences the extractive path will quote. Two is a
# deliberate cap, not a default: quoting the whole of every retrieved document
# is a copy operation that reads like an answer while answering nothing, and
# three sentences of unrelated evidence is harder for a reader to check than two.
_EXTRACTIVE_SENTENCE_BUDGET = 2

# A figure: a number with a magnitude, unit or currency, or a percentage. Used to
# prefer the sentence that carries a concrete value when several sentences tie on
# question-term overlap.
_FIGURE = re.compile(
    r"\d[\d,_.]*\s*(?:%|ms|s|m|h|kb|mb|gb|tb|kib|mib|gib|k|m|b|req|rps|qps|hz|°c|°f|x)?\b"
    r"|[$€£]\s*\d",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """What a generator produced, and what it cost.

    `degraded` is set when the requested generator could not run and a fallback
    answered instead. It is reported rather than raised, because an unconfigured
    provider is this project's normal state and an outage would misrepresent it.
    """

    text: str
    # `default_factory` rather than a shared `Usage()` instance. `Usage` is frozen
    # so the singleton default is currently harmless, but the safety would rest
    # on an invariant declared in another module; if it were ever unfrozen, every
    # default-constructed result would alias one mutable counter.
    usage: Usage = field(default_factory=Usage)
    strategy: str = "extractive"
    degraded: bool = False
    reason: str = ""


@runtime_checkable
class Generator(Protocol):
    """The interface `RagPipeline` depends on."""

    name: str

    def generate(self, question: str, context: ContextBlock) -> GenerationResult: ...


class ExtractiveGenerator:
    """Composes an answer by quoting the highest-ranked evidence.

    Ranks sentences by inverse-document-frequency-weighted overlap with the
    question's content words, rather than by document rank alone. A sentence in
    the third-ranked document that actually addresses the question beats
    boilerplate in the first, and ranking only by document order cannot express
    that.

    The weighting matters because a raw count prefers whichever sentence repeats
    the most *common* query words. Asking "What is the atlas-ingest availability
    objective?" against a single SLO document scores the latency paragraph
    higher than the availability one: it contains `atlas-ingest` and `objective`
    as well, so a plain count sees three matches against two and picks the wrong
    paragraph. `atlas-ingest` is in the document's title and therefore in every
    sentence that could be a candidate; it carries no information about which
    paragraph answers the question. Weighting by IDF over the retrieved context
    inverts that, so the one term that actually discriminates decides the match.
    """

    name = "extractive"

    def __init__(self, sentence_budget: int = _EXTRACTIVE_SENTENCE_BUDGET) -> None:
        self.sentence_budget = max(1, sentence_budget)

    def generate(self, question: str, context: ContextBlock) -> GenerationResult:
        chunks: Sequence[Chunk] = context.chunks
        if not chunks:
            return GenerationResult(
                text="", strategy=self.name, reason="no evidence to compose from"
            )

        asked = content_words(question)
        if not asked:
            return GenerationResult(
                text="", strategy=self.name, reason="the question has no content words to match"
            )

        # Two passes. The first collects the candidate units; the second scores
        # them. The order matters, because the weight of a query term depends on
        # how often it occurs among the candidates, and the candidates are not
        # known until they have been collected.
        candidates: list[tuple[int, str, set[str]]] = []
        seen: set[str] = set()
        for chunk in chunks:
            for position, sentence in enumerate(split_sentences(chunk.text)):
                if is_structural(sentence):
                    continue
                key = " ".join(sentence.lower().split())
                if key in seen:
                    continue
                seen.add(key)
                candidates.append((position, sentence, content_words(sentence)))

        if not candidates:
            return GenerationResult(
                text="", strategy=self.name, reason="the evidence contains no prose sentences"
            )

        # IDF over the candidate sentences -- the same population being ranked.
        #
        # Sentence granularity, not chunk or corpus. That is the detail that makes
        # this work: the competing sentences for "What is the atlas-ingest
        # availability objective?" are all inside a single SLO document, so
        # chunk-level statistics see them as identical and cannot prefer either.
        # `atlas-ingest` sits in the title of every retrieved chunk, so at chunk
        # level it still scored highly enough for the latency paragraph -- which
        # contains "atlas-ingest" and "objective" -- to outrank the availability
        # paragraph. Across sentences, `latency` recurs across the SLO, the
        # runbook and the legacy runbook while `availability` appears in one
        # place, and the term that actually discriminates wins.
        total = len(candidates)
        document_frequency = {
            term: sum(1 for _, _, words in candidates if term in words) for term in asked
        }

        # Every query term is scored, with IDF doing the weighting. A hard
        # high-frequency cutoff was tried here and removed: with a small
        # candidate pool it drops the term that actually answers the question.
        # For "What is the acme dedicated capacity?" the context holds 9 candidate
        # sentences and `capacity` occurs in 6 of them -- because the context is
        # five documents *about* capacity -- so a "more than half is
        # uninformative" rule classified the answer-bearing term as noise and
        # left only `acme` to score on, which selected a sentence about tenant
        # sign-off. Document frequency inside a topical context measures what the
        # context is about, not what distinguishes one sentence from another, and
        # a smooth weight handles the same case correctly.
        idf: dict[str, float] = {
            term: math.log1p(total / (1.0 + document_frequency[term])) + 0.25
            for term in asked
        }

        scored: list[tuple[float, float, int, int, str]] = []
        for position, sentence, words in candidates:
            matched = asked & words
            matched_idf = [idf[t] for t in matched]
            # The most distinctive query term matched is the primary signal, and
            # the IDF-weighted sum is only a tiebreak.
            #
            # Ranking on the sum alone is what made "What is the atlas-ingest
            # availability objective?" answer with the latency paragraph. Both
            # sentences contain `objective`; the latency one additionally contains
            # `atlas-ingest`, twice over as `atlas` and `ingest`, so its sum came
            # out ahead (5.79 vs 5.55) even though it is the one sentence in the
            # whole context that does not contain the word the question was
            # actually about. Counting how many query terms a sentence contains
            # rewards the sentence that echoes the question's most common words,
            # which is a measure of the question, not of the answer.
            #
            # Taking the maximum reverses it: `availability` occurs in one of the
            # 36 candidate sentences and so carries the highest weight in the
            # question, and the one sentence containing it wins. The sum then
            # separates sentences that match the same distinctive term, and
            # prefers the one that also matches more of the rest.
            #
            # This is a heuristic, tuned against the probe suite rather than
            # derived, so it is stated as one. Its known limit is lexical: a
            # question that paraphrases the evidence's own term scores on the
            # paraphrase instead. "What is the acme dedicated capacity?" retrieves
            # documents saying "contracted capacity", matches only on `acme`, and
            # returns a sentence about tenant sign-off. Confidence comes out low
            # (0.19), which is the signal that it should not be trusted; closing
            # the gap needs embeddings or a trained reranker, not a threshold
            # moved until the demo question passes.
            best_term = max(matched_idf) if matched_idf else 0.0
            total_weight = sum(matched_idf)
            # Figures break ties. A question asked about capacity, a limit or
            # a count is answered by the sentence containing the number, and
            # among sentences that share the same vocabulary the one with a
            # concrete value is the one that answers it. Position breaks
            # remaining ties toward earlier sentences, so a definition in the
            # opening of a document beats a closing remark.
            has_figure = 1 if _FIGURE.search(sentence) else 0
            scored.append((best_term, total_weight, has_figure, -position, sentence))

        scored.sort(key=lambda row: row[:4], reverse=True)
        best_key = scored[0][:2]

        # Only quote sentences that engage the question. Quoting the top-ranked
        # text regardless is how an extractive system produces a fluent answer to
        # a question it has no evidence for.
        #
        # The `best_overlap == 0` test is the one that matters. Every sentence ties
        # at zero when the question shares no content word with the evidence --
        # a paraphrase, or a question whose answer is a bare figure -- and an
        # earlier version of this compared every row to `best_overlap` and took
        # the first two, so that case returned the head of the document rather
        # than nothing. It answered "What is the acme dedicated capacity?" with
        # "Restricted to the acme tenant." Abstaining is the correct output when
        # nothing in the context addresses the question.
        if best_key[0] <= 0.0:
            return GenerationResult(
                text="",
                strategy=self.name,
                reason="no evidence sentence shares a content word with the question",
            )
        # Compare on a tolerance: these are sums of logs, and float equality would
        # drop a co-equal sentence whose IDF terms happened to sum a hair lower.
        relevant = [row[4] for row in scored if row[0] >= best_key[0] - 1e-9][
            : self.sentence_budget
        ]
        if not relevant:

            return GenerationResult(
                text="", strategy=self.name, reason="no evidence sentence addresses the question"
            )

        text = " ".join(relevant)
        prompt_chars = len(question) + len(context.text)
        return GenerationResult(
            text=text,
            usage=Usage(
                model="extractive",
                prompt_tokens=max(1, prompt_chars // 4),
                completion_tokens=max(1, len(text) // 4),
                latency_ms=0.0,
                estimated=True,
            ),
            strategy=self.name,
        )


class ProviderGenerator:
    """Calls an OpenAI-compatible chat endpoint and reports real token usage."""

    name = "provider"

    def __init__(self, settings: Settings, *, temperature: float = 0.0) -> None:
        self.settings = settings
        self.temperature = temperature

    def generate(self, question: str, context: ContextBlock) -> GenerationResult:
        try:
            text, usage = provider.complete(
                self.settings,
                format_messages(question, context),
                temperature=self.temperature,
            )
        except LLMUnavailable as exc:
            return GenerationResult(
                text="",
                strategy=self.name,
                degraded=True,
                reason=f"provider unavailable ({exc}); fell back to extraction",
            )
        return GenerationResult(
            text=text.strip(),
            usage=usage,
            strategy=self.name,
        )


def select_generator(settings: Settings) -> Generator:
    """The configured generator, or the deterministic default.

    Returning the extractive path when no provider is configured is what makes
    every published number in this repository reproducible with an empty
    environment. It is a real answer path, not a stub: the extractive generator
    is the one that produced the committed evaluation results.
    """
    if settings.llm_enabled:
        return ProviderGenerator(settings)
    return ExtractiveGenerator()
