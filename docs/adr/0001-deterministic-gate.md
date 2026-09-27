# ADR 0001 — Deterministic structure metrics as the regression gate

**Status:** accepted

## Context

A RAG pipeline needs a quality number to compare runs. The available options were
an LLM-as-judge score, embedding similarity to a reference answer, or structural
metrics computed from the retrieved set.

The first two are not reproducible. A judge score changes when the judge model is
rebuilt, when a provider changes a system prompt, or when a temperature is not
pinned. Embedding similarity needs a hosted model to compare against.

Neither can serve as a regression gate, and that is the whole problem: a gate
that produces a different number on an unchanged system is indistinguishable from
a gate that is broken.

## Decision

The gating suite uses only structure that is already in the retrieval result:
which sources were retrieved, at what rank, and which fact strings appear in the
text. No model is called.

`pass_rate` is the headline. A probe passes when every obligation it declares is
satisfied. Losing an obligation is a failure even if the answer text looks right,
because the retrieved set is the only thing the pipeline controls.

LLM-as-judge scoring is kept for exploration in a **separate, explicitly
non-gating suite** (see ADR 0003).

## Consequences

- Every number in `README.md` is reproducible offline with no credentials and no
  network.
- The suite cannot measure whether retrieved evidence was *relevant* to a
  question the probes did not anticipate. That is a real blind spot, and it is
  why `context_precision` — which can detect over-retrieval — is reported even
  though it is low at 0.242.
- Fact extraction from the reference answers is manual. At 90 probes that is
  sustainable; it does not scale to 90,000, and a corpus of that size needs a
  different gate.
