"""Load a corpus from disk into `Chunk` objects.

Deliberately separate from the Qdrant ingest path. The benchmark needs a
labelled, in-memory corpus that is byte-identical on every run and on every
machine; a vector store is the wrong tool for that, and making the benchmark
depend on a running Qdrant would mean the headline numbers in the README could
not be reproduced by a reader with no infrastructure.

It also gives the retrieval package a single documented entry point, instead of
every caller re-implementing directory walking, extension filtering and
stable-ID assignment slightly differently.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Iterable, Sequence

from enterprise_rag.config.settings import Settings
from enterprise_rag.indexing.manifest import parse_front_matter
from enterprise_rag.chunking.splitter import chunk_text
from enterprise_rag.types import Chunk, stable_chunk_id

__all__ = ["Corpus", "load_corpus", "SOURCE_EXTENSIONS"]

SOURCE_EXTENSIONS = (".md", ".markdown", ".txt", ".rst")

# Document-level keys copied onto every chunk. These are the keys a filter needs
# to make a decision, so they must be on the chunk rather than only on a
# sidecar: once chunks are in a vector store, the sidecar is a lookup that
# silently disappears under a partial re-index, and a filter that silently stops
# filtering is worse than one that loudly fails.
PROPAGATED_METADATA = ("tenant", "status", "version", "updated", "owner", "acl", "supersedes")


@dataclass(frozen=True, slots=True)
class Corpus:
    """An immutable, in-memory document collection."""

    chunks: tuple[Chunk, ...]
    sources: tuple[str, ...]
    root: str

    def __len__(self) -> int:
        return len(self.chunks)

    def __iter__(self):
        return iter(self.chunks)

    def scoped(self, allowed: Sequence[str] | None = None) -> "Corpus":
        """Restrict to a set of `source_type` values.

        The scope filter is applied at *retrieval* time as well as here, because
        a Qdrant-backed deployment can only filter by a payload index, not by
        post-filtering in Python. Doing it in both places keeps the in-memory
        benchmark path and the production path semantically identical â€” if the
        filter only existed locally, the benchmark would measure a system that
        does not ship.
        """
        if not allowed:
            return self
        # A list, not a set: `Chunk` carries a `metadata` dict and is therefore
        # unhashable, so a set comprehension over chunks raises TypeError.
        permitted = set(allowed)
        keep = [c for c in self.chunks if c.source_type in permitted]
        sources = tuple(dict.fromkeys(c.source for c in keep))
        return Corpus(chunks=tuple(keep), sources=sources, root=self.root)

    def by_id(self, chunk_id: str) -> Chunk | None:
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    def ids_for_source(self, source: str) -> tuple[str, ...]:
        return tuple(c.chunk_id for c in self.chunks if c.source == source)

    @property
    def tenants(self) -> frozenset[str]:
        return frozenset(c.metadata.get("tenant", "public") for c in self.chunks)

    def visible_to(self, allowed: Sequence[str]) -> "Corpus":
        """Restrict to chunks whose document tenant is in `allowed`.

        The access-control primitive. Three properties are load-bearing:

        * It fails closed. A chunk with no tenant is treated as non-public and is
          only returned when `"public"` is explicitly allowed, so a document that
          lost its front matter during a bad export becomes invisible rather than
          becoming world-readable.
        * It filters on *document* tenancy, carried onto every chunk at load
          time, so there is no sidecar join that a partial re-index can drop.
        * It returns a `Corpus` rather than a filtered list, so a caller cannot
          accidentally score a leak against a chunk set they never filtered.

        The same filter must be applied by the vector-store path via a payload
        index, not by post-filtering here. Post-filtering is only equivalent when
        the candidate pool is complete, and with `candidate_k=30` a restricted
        document outside the pool is invisible to a post-filter and in clear view
        to a user. See docs/adr/0005.
        """
        permitted = set(allowed)
        if not permitted:
            # An empty allow-list means "no authorisation was established", which
            # must read as "show nothing". Treating it as "show everything" is the
            # classic fail-open bug in document APIs.
            return Corpus(chunks=(), sources=(), root=self.root)

        keep = [c for c in self.chunks if c.metadata.get("tenant", "public") in permitted]
        if not keep:
            return Corpus(chunks=(), sources=(), root=self.root)
        sources = tuple(dict.fromkeys(c.source for c in keep))
        return Corpus(chunks=tuple(keep), sources=sources, root=self.root)

    def by_status(self, *statuses: str) -> "Corpus":
        """Restrict to chunks whose document status is one of `statuses`.

        Used to drop superseded revisions from the candidate pool. Filtering the
        corpus is the honest way to do this; demoting the score is not, because
        a demotion that is too small loses to a lexical match and a demotion that
        is too large discards genuinely relevant drafts.
        """
        wanted = {s.lower() for s in statuses}
        keep = [c for c in self.chunks if c.metadata.get("status", "").lower() in wanted]
        sources = tuple(dict.fromkeys(c.source for c in keep))
        return Corpus(chunks=tuple(keep), sources=sources, root=self.root)


def load_corpus(
    root: str | pathlib.Path,
    *,
    source_type: str = "true",
    max_chars: int = 1200,
    overlap: int = 120,
    extensions: Iterable[str] = SOURCE_EXTENSIONS,
) -> Corpus:
    """Read, chunk and ID every document under `root`.

    IDs are content-derived (`stable_chunk_id`), not random, so re-ingesting an
    unchanged corpus produces identical IDs. The previous ingest path used
    `uuid4()` per chunk, which meant every run looked like a brand new corpus
    to Qdrant: unchanged documents were re-embedded, re-upserted and re-paid
    for, and there was no way to tell an edit from a delete-and-recreate.

    Known trade-off, stated rather than hidden: the ID mixes in the chunk's
    ordinal within the document, so inserting a paragraph near the top shifts
    every later chunk's ID and they are all re-embedded. The alternative
    (hashing the text alone) would be stable under insertion but would collide
    when a document legitimately repeats a chunk. For a doc store where documents
    are edited occasionally and mostly appended to, occasional re-embedding of
    one document is the cheaper mistake.
    """
    root_path = pathlib.Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"corpus root does not exist: {root_path}")

    allowed_ext = {e.lower() for e in extensions}
    chunks: list[Chunk] = []

    files = sorted(
        p for p in root_path.rglob("*") if p.is_file() and p.suffix.lower() in allowed_ext
    )
    ordinal = 0
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            continue
        source = path.relative_to(root_path).as_posix()
        front, body = parse_front_matter(text)

        # `tenant` defaults to "public" rather than being absent, so a filter
        # written as "keep tenants in {allowed}" cannot accidentally treat a
        # missing value as a match. Absent metadata must fail closed.
        document_metadata = {"tenant": "public"}
        for key in PROPAGATED_METADATA:
            if front.get(key):
                document_metadata[key] = front[key]

        for piece in chunk_text(body, chunk_size=max_chars, overlap=overlap):
            cleaned = piece.strip()
            if not cleaned:
                continue
            metadata = dict(document_metadata)
            metadata["bytes"] = str(len(cleaned))
            chunks.append(
                Chunk(
                    chunk_id=stable_chunk_id(source, ordinal, cleaned),
                    text=cleaned,
                    source=source,
                    source_type=source_type,
                    ordinal=ordinal,
                    metadata=metadata,
                )
            )
            ordinal += 1

    return Corpus(
        chunks=tuple(chunks),
        sources=tuple(dict.fromkeys(c.source for c in chunks)),
        root=str(root_path),
    )
