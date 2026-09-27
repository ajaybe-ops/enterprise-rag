"""Production-grade Enterprise Retrieval-Augmented Generation.

A retrieval system built around one commitment: **every answer must be
attributable, and every quality number must be explainable.**

Naive RAG answers a question and returns text. When that text is wrong there is
no way to tell whether the document was never retrieved, retrieved and outranked,
retrieved but truncated away, or retrieved and then ignored by the generator.
Those are four different bugs with four different fixes, and they all look
identical from the outside.

This package makes each of them observable:

- the corpus is a content-addressed artifact (:mod:`.indexing`), so a metric
  movement can be attributed to content rather than to code;
- retrieval is hybrid and instrumented (:mod:`.retrieval`), so a miss is
  attributable to a specific leg;
- context construction is a budgeted, ordered stage (:mod:`.context`), not a
  string join;
- generation is separable from verification (:mod:`.generation`,
  :mod:`.citations`), so an unsupported sentence is dropped rather than served;
- evaluation scores retrieval structure, not an LLM's opinion of itself
  (:mod:`.evaluation`), so a regression gate is reproducible.

The architecture is deliberately a single pipeline. Multi-agent retrieval is not
implemented and is not planned: agent-to-agent fan-out multiplies latency and
cost per query, makes attribution impossible (which agent's evidence produced
this claim?), and buys nothing that a well-tuned hybrid retriever does not. The
interesting engineering problem in RAG is reliability, not orchestration.

See ``docs/architecture.md`` for the stage-by-stage breakdown and
``docs/failure-modes.md`` for what happens when each dependency fails.
"""

from __future__ import annotations

__version__ = "0.1.0"

# The package layout, in dependency order. Exported as data so that
# documentation, ``/health`` and the packaging metadata cannot drift from the
# directory tree: a stage list that is maintained in three places is a stage
# list that is wrong in two of them.
STAGES: tuple[str, ...] = (
    "ingestion",
    "parsing",
    "chunking",
    "indexing",
    "retrieval",
    "reranking",
    "context",
    "generation",
    "citations",
    "security",
    "evaluation",
    "observability",
)

__all__ = ["STAGES", "__version__"]
