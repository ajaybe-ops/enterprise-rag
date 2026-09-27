"""Content-addressed corpus manifests.

Why this exists
---------------
A retrieval system that serves a changing corpus degrades *quietly*. Someone
edits a document, chunk boundaries shift, and the evidence a query needed now
sits at rank 9 instead of rank 2. Nothing crashes. Aggregate recall may not even
move, because the corpus grew a hundred other chunks in the same commit.

The only way to tell "retrieval got worse" apart from "the corpus changed under
us" is to make the corpus a *versioned artifact* rather than a directory. That
means every document needs a stable identity, a content hash, and a recorded
chunk decomposition, so two runs can be diffed structurally.

This module is that substrate. It answers one question precisely:

    what changed about the corpus between run A and run B?

Everything in `app.evaluation.regression` that claims to *explain* a regression
is built on the diff produced here. Without it, "recall dropped" is a shrug.

Design notes
------------
Hashes are content hashes of the *normalised document text*, not of the bytes on
disk. A git checkout that rewrites line endings, or an editor that changes
trailing whitespace, must not read as a document edit -- otherwise every
cross-platform diff is noise and the signal is ignored.

Chunk identity deliberately includes the ordinal (see `stable_chunk_id`), which
means an insertion near the top of a document invalidates every later chunk id.
That is a real cost, paid deliberately: it is what lets the diff distinguish
"this document was edited" from "this document was re-chunked differently", and
the second is a far more common cause of silent retrieval regressions than
content edits are. See docs/adr/0002.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping, Sequence

__all__ = [
    "DocumentRecord",
    "CorpusManifest",
    "ManifestDiff",
    "ChangedDocument",
    "build_manifest",
    "parse_front_matter",
    "normalise_text",
    "content_hash",
    "MANIFEST_VERSION",
    "FRONT_MATTER_KEYS",
]

MANIFEST_VERSION = 1

# The metadata an enterprise corpus needs in order to be *filterable*: tenancy for
# access control, status for staleness, and ownership for routing. This list is
# short and explicit on purpose. A general YAML parse here would make the shape
# of the index depend on whatever a document happens to contain, and a security
# control should depend on code, not on document formatting.
FRONT_MATTER_KEYS = ("tenant", "status", "version", "updated", "owner", "acl", "supersedes")

# Trailing whitespace is invisible in review but changes every byte-level hash,
# so it is collapsed before hashing. Line endings are normalised by the reader
# (see `normalise_text`), not here.
_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)


def normalise_text(text: str) -> str:
    """Canonical form used for both hashing and display.

    Normalising CRLF and trailing whitespace before hashing means a manifest
    built on Windows compares equal to one built on Linux. Without this, the
    same corpus produces two different manifest ids on two platforms and every
    cross-machine experiment looks like a total rewrite.
    """
    return _TRAILING_WS.sub("", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def content_hash(text: str) -> str:
    """SHA-256 of the normalised text, truncated for readability in reports.

    16 hex chars is 64 bits: collision probability stays negligible for corpora
    far larger than anything this system will hold in memory, and a full digest
    makes diff reports unreadable.
    """
    return hashlib.sha256(normalise_text(text).encode("utf-8")).hexdigest()[:16]


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split `---` front matter from the body. Returns (metadata, body).

    Intentionally not YAML. The keys are a fixed allow-list of flat scalars, and
    a restricted parser cannot be surprised by an alias bomb, a custom tag, or a
    nested structure that silently changes a filter's meaning. A document that
    wants richer metadata gets a sidecar, not a richer parser in the hot path.
    """
    if not text.startswith("---"):
        return {}, text

    end = text.find("\n---", 3)
    if end == -1:
        return {}, text

    header = text[3:end]
    body = text[end + 4 :].lstrip("\n")
    metadata: dict[str, str] = {}
    for line in header.splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        if key in FRONT_MATTER_KEYS:
            metadata[key] = value.strip().strip("\"'")
    return metadata, body


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """One document as it existed at one point in time.

    `doc_id` is derived from the *path*, not the content. That is a deliberate
    inversion of the usual instinct: if identity followed content, editing a
    document would retire one document and create another, and the diff could
    never say "this document changed" -- only "one document vanished and an
    unrelated one appeared". Path-derived identity is what makes edit detection
    possible at all.
    """

    doc_id: str
    source: str
    content_hash: str
    char_count: int
    chunk_ids: tuple[str, ...]
    metadata: Mapping[str, str] = field(default_factory=dict)

    @property
    def chunk_count(self) -> int:
        return len(self.chunk_ids)

    @property
    def source_type(self) -> str:
        return self.metadata.get("source_type", "true")

    @property
    def tenant(self) -> str:
        """Owning tenant, defaulting to the shared `public` scope.

        Defaulting to `public` is a deliberate fail-*open* choice and the one
        place in this module where that is true. It exists so a corpus with no
        front matter at all -- a quick `load_corpus("docs/")` against a plain
        directory of notes -- is still readable, which is the common case while
        exploring. The cost is that forgetting `tenant:` on a restricted document
        publishes it.

        The safer alternative (treat a missing tenant as invisible unless `public`
        is explicitly requested) was rejected because it makes the default corpus
        silently empty, and an empty result is harder to notice than a wrong one.
        Ingest therefore writes `tenant` explicitly for every document it creates
        (`app/ingestion/ingest.py`), and a production corpus should carry
        `tenant:` in every file. `tests/test_security.py` pins the behaviour of
        both branches so the trade-off cannot be changed silently.
        """
        return self.metadata.get("tenant", "public")

    @property
    def status(self) -> str:
        """Lifecycle status (`current`, `draft`, `superseded`). Defaults to `current`.

        Defaults to `current` for the same reason `tenant` defaults to `public`:
        a document that says nothing about its lifecycle is treated as live. The
        asymmetry is that this is fail-*open* towards being served, which is the
        safe direction for a stale document -- a `superseded` doc that is wrongly
        served is confusing; a `current` doc that is wrongly withheld is a
        support ticket. A document that must never be served must say
        `status: superseded` explicitly.
        """
        return self.metadata.get("status", "current")

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "content_hash": self.content_hash,
            "char_count": self.char_count,
            "chunk_ids": list(self.chunk_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "DocumentRecord":
        return cls(
            doc_id=payload["doc_id"],
            source=payload["source"],
            content_hash=payload["content_hash"],
            char_count=int(payload.get("char_count", 0)),
            chunk_ids=tuple(payload.get("chunk_ids", ())),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class ChangedDocument:
    """A document that differs between two manifests."""

    source: str
    kind: str  # "added" | "removed" | "modified"
    before: DocumentRecord | None = None
    after: DocumentRecord | None = None

    @property
    def chunk_delta(self) -> int:
        """Change in chunk count. Non-zero with an unchanged content hash means
        the *chunker* changed, not the document -- a distinction that matters,
        because it implicates the pipeline rather than the corpus."""
        before = self.before.chunk_count if self.before else 0
        after = self.after.chunk_count if self.after else 0
        return after - before

    @property
    def content_changed(self) -> bool:
        if self.before is None or self.after is None:
            return True
        return self.before.content_hash != self.after.content_hash

    @property
    def rechunked(self) -> bool:
        """Same text, different chunk decomposition."""
        return (
            self.before is not None
            and self.after is not None
            and not self.content_changed
            and self.before.chunk_ids != self.after.chunk_ids
        )

    def explain(self) -> str:
        if self.kind == "added":
            return f"{self.source}: added ({self.after.chunk_count if self.after else 0} chunks)"
        if self.kind == "removed":
            return f"{self.source}: removed ({self.before.chunk_count if self.before else 0} chunks)"
        if self.rechunked:
            return (
                f"{self.source}: re-chunked, text unchanged "
                f"({self.before.chunk_count}->{self.after.chunk_count} chunks)"
                if self.before and self.after
                else f"{self.source}: re-chunked"
            )
        return (
            f"{self.source}: edited "
            f"({self.before.char_count if self.before else 0}->"
            f"{self.after.char_count if self.after else 0} chars, "
            f"chunks {self.before.chunk_count if self.before else 0}->"
            f"{self.after.chunk_count if self.after else 0})"
            if self.before and self.after
            else f"{self.source}: edited"
        )


@dataclass(frozen=True, slots=True)
class ManifestDiff:
    """Structural difference between two corpus versions."""

    added: tuple[ChangedDocument, ...] = ()
    removed: tuple[ChangedDocument, ...] = ()
    modified: tuple[ChangedDocument, ...] = ()
    unchanged_count: int = 0

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.modified)

    @property
    def changed_sources(self) -> frozenset[str]:
        return frozenset(c.source for c in (*self.added, *self.removed, *self.modified))

    @property
    def rechunked(self) -> tuple[ChangedDocument, ...]:
        return tuple(c for c in self.modified if c.rechunked)

    def summary(self) -> str:
        if self.is_empty:
            return "corpus unchanged"
        parts = []
        if self.added:
            parts.append(f"+{len(self.added)} added")
        if self.removed:
            parts.append(f"-{len(self.removed)} removed")
        if self.modified:
            parts.append(f"~{len(self.modified)} modified")
        return ", ".join(parts)

    def to_dict(self) -> dict:
        return {
            "added": [c.explain() for c in self.added],
            "removed": [c.explain() for c in self.removed],
            "modified": [c.explain() for c in self.modified],
            "unchanged_count": self.unchanged_count,
            "is_empty": self.is_empty,
        }


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    """A content-addressed snapshot of an entire corpus.

    `manifest_id` is a hash of the sorted (source, content_hash) pairs. Two
    manifests with the same id are byte-identical corpora, which is what lets an
    experiment record "measured against corpus X" and have that claim mean
    something later.
    """

    records: tuple[DocumentRecord, ...]
    manifest_id: str
    corpus_root: str = ""
    version: int = MANIFEST_VERSION
    notes: str = ""

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(r.source for r in self.records)

    @property
    def total_chunks(self) -> int:
        return sum(r.chunk_count for r in self.records)

    def by_source(self, source: str) -> DocumentRecord | None:
        for record in self.records:
            if record.source == source:
                return record
        return None

    def all_chunk_ids(self) -> frozenset[str]:
        return frozenset(cid for record in self.records for cid in record.chunk_ids)

    def diff(self, other: "CorpusManifest") -> ManifestDiff:
        """Structural diff against `other` (the newer manifest)."""
        before = {r.source: r for r in self.records}
        after = {r.source: r for r in other.records}

        added, removed, modified, unchanged = [], [], [], 0
        for source, new in after.items():
            old = before.get(source)
            if old is None:
                added.append(ChangedDocument(source=source, kind="added", after=new))
            elif old.content_hash != new.content_hash or old.chunk_ids != new.chunk_ids:
                modified.append(ChangedDocument(source=source, kind="modified", before=old, after=new))
            else:
                unchanged += 1
        for source, old in before.items():
            if source not in after:
                removed.append(ChangedDocument(source=source, kind="removed", before=old))

        key = lambda c: c.source  # noqa: E731 - local sort key
        return ManifestDiff(
            added=tuple(sorted(added, key=key)),
            removed=tuple(sorted(removed, key=key)),
            modified=tuple(sorted(modified, key=key)),
            unchanged_count=unchanged,
        )

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "manifest_id": self.manifest_id,
            "corpus_root": self.corpus_root,
            "notes": self.notes,
            "document_count": len(self.records),
            "total_chunks": self.total_chunks,
            "records": [r.to_dict() for r in self.records],
        }

    def save(self, path: str | pathlib.Path) -> pathlib.Path:
        """Write the manifest as deterministic JSON.

        Sorted keys and a trailing newline: manifests get committed and diffed
        in review, so byte-stable output is a feature, not a detail.
        """
        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "CorpusManifest":
        payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        return cls(
            records=tuple(DocumentRecord.from_dict(r) for r in payload.get("records", ())),
            manifest_id=payload.get("manifest_id", ""),
            corpus_root=payload.get("corpus_root", ""),
            version=int(payload.get("version", MANIFEST_VERSION)),
            notes=payload.get("notes", ""),
        )


def _manifest_id(records: Sequence[DocumentRecord]) -> str:
    """Hash of (source, content_hash, chunk_ids) triples.

    Includes chunk ids, not just content hashes, so a chunker change produces a
    different manifest id even when no document was edited. Two corpora that
    produce different retrievals must not share an id.
    """
    material = "\n".join(
        f"{r.source}:{r.content_hash}:{','.join(r.chunk_ids)}"
        for r in sorted(records, key=lambda r: r.source)
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def build_manifest(
    root: str | pathlib.Path,
    *,
    extensions: Iterable[str] = (".md", ".markdown", ".txt", ".rst"),
    source_type: str = "true",
    tenant: str = "public",
    default_metadata: Mapping[str, str] | None = None,
) -> CorpusManifest:
    """Walk `root` and build a manifest from the files on disk.

    Reads whole files rather than reusing `load_corpus`, because the manifest is
    a property of the *documents*, and deriving it from chunks would mean the
    document hash changed every time the chunker did -- destroying the ability to
    distinguish "edited" from "re-chunked", which is the entire reason
    `ChangedDocument.rechunked` exists.
    """
    root_path = pathlib.Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"corpus root does not exist: {root_path}")

    allowed = {e.lower() for e in extensions}
    records: list[DocumentRecord] = []
    base_metadata = {"source_type": source_type, "tenant": tenant}
    if default_metadata:
        base_metadata.update(default_metadata)

    for path in sorted(p for p in root_path.rglob("*") if p.is_file() and p.suffix.lower() in allowed):
        raw = path.read_text(encoding="utf-8", errors="replace")
        if not raw.strip():
            continue
        text = normalise_text(raw)
        source = path.relative_to(root_path).as_posix()

        metadata = dict(base_metadata)
        front, _ = parse_front_matter(raw)
        metadata.update(front)

        records.append(
            DocumentRecord(
                doc_id=hashlib.sha256(f"doc:{source}".encode("utf-8")).hexdigest()[:16],
                source=source,
                content_hash=content_hash(text),
                char_count=len(text),
                # Left empty on purpose: populating this requires the chunker,
                # and a manifest that mixes document state with chunk state
                # cannot tell the two apart. `attach_chunks` fills it in.
                chunk_ids=(),
                metadata=metadata,
            )
        )

    records.sort(key=lambda r: r.source)
    return CorpusManifest(
        records=tuple(records),
        manifest_id=_manifest_id(records),
        corpus_root=str(root_path),
    )


def attach_chunks(manifest: CorpusManifest, chunks: Sequence) -> CorpusManifest:
    """Return a copy of `manifest` with chunk ids recorded per document.

    Kept separate from `build_manifest` so the document-level hash is computed
    from the document alone. Callers that only want to know "did the text
    change" never pay for chunking, and the two failure modes stay separable.
    """
    grouped: dict[str, list[str]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.source, []).append(chunk.chunk_id)

    records = tuple(
        replace(record, chunk_ids=tuple(grouped.get(record.source, ())))
        for record in manifest.records
    )
    return replace(
        manifest,
        records=records,
        manifest_id=_manifest_id(records),
    )
