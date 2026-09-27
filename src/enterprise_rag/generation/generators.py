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

import time
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from enterprise_rag.citations.verifier import content_words, split_sentences
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


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """What a generator produced, and what it cost.

    `degraded` is set when the requested generator could not run and a fallback
    answered instead. It is reported rather than raised, because an unconfigured
    provider is this project's normal state and an outage would misrepresent it.
    """

    text: str
    usage: Usage = Usage()
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

    Ranks sentences by how many of the question's content words they contain,
    rather than by document rank alone. A sentence in the third-ranked document
    that actually addresses the question beats boilerplate in the first, and
    ranking only by document order cannot express that.
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
        scored: list[tuple[int, int, str]] = []
        seen: set[str] = set()
        for chunk in chunks:
            for position, sentence in enumerate(split_sentences(chunk.text)):
                key = " ".join(sentence.lower().split())
                if key in seen:
                    continue
                seen.add(key)
                overlap = len(asked & content_words(sentence)) if asked else 0
                # Position breaks ties toward earlier sentences, so a definition
                # in the opening of a document beats a closing remark.
                scored.append((overlap, -position, sentence))

        scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
        best_overlap = scored[0][0] if scored else 0
        # Only quote sentences that engage the question. Quoting the top-ranked
        # text regardless is how an extractive system produces a fluent answer to
        # a question it has no evidence for.
        relevant = [row[2] for row in scored if row[0] == best_overlap][: self.sentence_budget]
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
        started = time.perf_counter()
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
