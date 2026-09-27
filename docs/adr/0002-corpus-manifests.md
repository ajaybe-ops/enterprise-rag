# ADR 0002 — The corpus is a content-addressed artifact

**Status:** accepted

Referenced from `app/corpus/manifest.py`.

## Context

When a metric moves, three different things could have changed: the corpus
content, the chunking, or the retrieval code. Reporting "recall fell by 3%" is
not actionable without knowing which.

The original pipeline indexed into a vector database. That made the corpus an
effect rather than an artifact: there was no durable record of which documents
a given answer was drawn from, so a regression could not be attributed to
content. It also made deduplication process-local — the registry lived in
memory, so a restart re-indexed everything.

## Decision

The corpus is identified by a manifest built from the files themselves.

- Each document is hashed over normalised text, so CRLF and trailing whitespace
  do not make the same corpus look different on Windows and Linux.
- Front matter is parsed into metadata and carried onto every chunk.
- Chunk ids are content-derived (`stable_chunk_id`), not assigned.
- The manifest id folds in **chunk identity, not chunk count**.

That last point is the load-bearing one. Hashing the count makes a rechunk
invisible: split one chunk into two and every later ordinal shifts, but the count
is unchanged, so the manifest id is unchanged and the diff reports "unchanged".
A retrieval regression caused entirely by chunking would then be attributed to
code. `tests/test_manifest.py::test_manifest_id_moves_when_chunk_identity_changes`
pins this.

## Consequences

- A metric movement is attributable: `corpus_edit`, `config_change`, or neither.
- The manifest id moves when the *chunker* changes even if no document was
  edited. That is intended, but it means the id is not a pure content address —
  it identifies "this corpus under this decomposition".
- `attach_chunks` is separate from `build_manifest` so a caller who only wants to
  know whether text changed does not pay for chunking.
