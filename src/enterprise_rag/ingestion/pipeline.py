"""Ingestion: files on disk â†’ text â†’ a versioned corpus.

Replaces the previous `processor.py`, which parsed documents correctly and then
threw the result away into a Qdrant upsert guarded by an in-process registry.
Two things were wrong with that shape, and both were structural rather than
stylistic:

* **The corpus was not an artifact.** Indexing produced side effects in a
  database. Two runs over the same inputs could differ, nothing recorded *which*
  documents a given answer was drawn from, and "which corpus produced this
  regression?" had no answer. `app/corpus/manifest.py` makes the corpus a
  content-addressed artifact, and this module writes into that.
* **Deduplication was process-local and lost on exit.** The registry lived in
  memory, so a restart re-indexed everything and two processes could not
  coordinate. `build_manifest` hashes content, so skipping an unchanged document
  is a property of the data on disk, not of a running process.

The Qdrant path was not ported. This reference build indexes into the manifest
and runs hybrid retrieval in-process; a hosted vector store is a deployment
choice, and reintroducing one here would add a dependency that cannot be
exercised offline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from enterprise_rag.observability import tracing as logfire  # vendor-free tracing shim
from enterprise_rag.indexing.manifest import build_manifest, normalise_text
from enterprise_rag.parsing.html import parse_html
from enterprise_rag.parsing.office import parse_office
from enterprise_rag.parsing.pdf import parse_pdf
from enterprise_rag.parsing.text import parse_text

__all__ = ["ExtractedDocument", "extract", "ingest_file", "ingest_directory", "SUPPORTED"]

SUPPORTED = frozenset({".pdf", ".docx", ".pptx", ".html", ".htm", ".md", ".txt", ".text"})

_PARSERS = (
    (frozenset({".pdf"}), parse_pdf),
    (frozenset({".docx", ".pptx"}), parse_office),
    (frozenset({".html", ".htm"}), parse_html),
)


@dataclass
class ExtractedDocument:
    """A parsed file plus the metadata that must travel with it.

    `metadata` is carried through to every chunk at load time rather than being
    re-derived at query time. Access control that is reconstructed from context
    at query time is access control that a future code path can forget to
    reconstruct, and the failure mode is a data leak.
    """

    source: str
    text: str
    content_hash: str
    metadata: dict

    @property
    def chars(self) -> int:
        return len(self.text)


def extract(path: str | os.PathLike[str], **metadata: object) -> ExtractedDocument | None:
    """Parse one file to normalised text. `None` if unsupported or empty.

    Returns `None` rather than raising on an unreadable or empty document. An
    ingest run over a directory of heterogeneous files should not abort because
    one file is a scanned PDF with no text layer -- the useful behaviour is to
    ingest the rest and report the omission, which is what `ingest_directory`
    does.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    text = ""
    for suffixes, parser in _PARSERS:
        if suffix in suffixes:
            try:
                text = parser(str(path))
            except Exception as exc:  # a corrupt file must not stop the run
                logfire.warning("parse failed", source=path.name, error=str(exc))
                return None
            break
    else:
        if suffix not in SUPPORTED:
            return None
        text = parse_text(str(path))

    text = normalise_text(text)
    if not text:
        return None

    from enterprise_rag.indexing.manifest import content_hash

    return ExtractedDocument(
        source=path.name,
        text=text,
        content_hash=content_hash(text),
        metadata={"source_type": suffix.lstrip("."), **metadata},
    )


def _front_matter(fields: dict) -> str:
    """Render a metadata header.

    Values are emitted in sorted order so that re-ingesting an unchanged document
    produces a byte-identical file. That is what lets `build_manifest` treat
    "no change" as no change: without deterministic ordering, a rewrite would
    change the content hash of a document nobody had edited.
    """
    lines = [f"{key}: {value}" for key, value in sorted(fields.items()) if value is not None]
    return "---\n" + "\n".join(lines) + "\n---\n\n"


def ingest_file(
    source_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    **metadata: object,
) -> str | None:
    """Parse one file and write it into the corpus directory as Markdown.

    Returns the output filename, or `None` if the file yielded no text.
    """
    document = extract(source_path, **metadata)
    if document is None:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = Path(document.source).stem + ".md"
    (out_dir / out_name).write_text(
        _front_matter(document.metadata) + document.text, encoding="utf-8"
    )
    return out_name


def ingest_directory(
    in_dir: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    **metadata: object,
) -> dict:
    """Ingest every supported file in `in_dir`. Returns a small report.

    `ingest_directory` writes files, and every other stage in this project reads
    a manifest rather than a database. A filesystem is not a database: it has no
    transactions, so a run interrupted halfway leaves a partially-populated
    corpus. That is acceptable here because `build_manifest` is the source of
    truth and can be re-run at any time -- the directory is a cache of parsed
    text, not the corpus.
    """
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written, skipped = [], []
    for path in sorted(in_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        name = ingest_file(path, out_dir, **metadata)
        (written if name else skipped).append(path.name)

    manifest = build_manifest(out_dir)
    return {
        "written": written,
        "skipped": skipped,
        "documents": len(manifest.documents),
        "manifest_id": manifest.manifest_id,
    }
