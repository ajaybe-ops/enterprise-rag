# ADR 0004 — Dense retrieval is LSA over numpy arrays, not an embedding service

**Status:** accepted

Referenced from `app/retrieval/bm25.py` and `app/retrieval/pipeline.py`.

## Context

Hybrid retrieval needs a dense leg. The options were a hosted embedding API
(Jina), a local sentence-transformer, or something that works with neither.

A hosted API makes every score non-reproducible: the model changes under you, the
network can fail, and the results cannot be regenerated in a year. A local
transformer is a multi-hundred-megabyte dependency that pins a torch version,
which makes environment resolution painful — this environment currently has
`opencv-python 4.13` requiring `numpy>=2` against `numpy 1.26.4`, and adding torch
makes that worse rather than better.

## Decision

The dense leg defaults to a truncated SVD (latent semantic analysis) over a
hashed bag-of-words matrix, computed in-process with numpy. No network, no
model download, deterministic for a fixed corpus.

`/health` reports the active retriever and any degradation, because a dense leg
that silently fell back must not look like a healthy system.

A real embedding model is supported and is the correct choice at scale. It is
opt-in precisely because it makes scores non-reproducible, and it must then be
recorded in the manifest.

## Consequences

- Retrieval is fully offline and reproducible, and the ablation suite can run in
  seconds per variant.
- The dense leg is weak. On the current corpus `dense-only` scores the same
  1.000 pass rate as `bm25-only` at 319ms versus 16ms — the LSA projection is
  buying nothing measurable while costing 20× the latency. That is the honest
  result, and the recommendation is to drop the dense leg until a real embedding
  model is evaluated.
- "Dense" here does not mean semantic. It means a linear projection of lexical
  co-occurrence. Paraphrase probes are the honest test of that, and they pass —
  but on this corpus, largely because documents are lexically distinct.
