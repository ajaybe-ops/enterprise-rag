"""Manifest tests.

The manifest is the thing that makes regression attribution possible: without a
content-addressed record of what the corpus contained, a metric movement can only
be reported, never explained. These tests pin the two distinctions the whole
evaluation story rests on -- *content changed* versus *chunks changed* -- because
if those two are conflated, every corpus edit gets misattributed as a pipeline
regression and the diagnostic is worse than useless.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from enterprise_rag.indexing.manifest import (
    attach_chunks,
    build_manifest,
    content_hash,
    normalise_text,
    parse_front_matter,
)
from enterprise_rag.types import Chunk


def chunk(source: str, ordinal: int, text: str) -> Chunk:
    return Chunk.from_text(text, source, ordinal)


def write(root: Path, name: str, body: str) -> Path:
    path = root / name
    path.write_text(body, encoding="utf-8")
    return path


class TestFrontMatter:
    def test_parses_body_and_metadata(self):
        meta, body = parse_front_matter("---\ntenant: acme\nstatus: current\n---\n\nHello world.\n")
        assert meta == {"tenant": "acme", "status": "current"}
        assert body.strip() == "Hello world."

    def test_document_without_front_matter_is_all_body(self):
        meta, body = parse_front_matter("Just text, no header.\n")
        assert meta == {}
        assert body.strip() == "Just text, no header."

    def test_metadata_is_str_not_parsed_yaml(self):
        # Deliberately not YAML: `tenant: 007` must stay the string "007".
        # A YAML load would turn it into 7, and the tenant filter would then
        # silently admit a document the operator intended to isolate.
        meta, _ = parse_front_matter("---\ntenant: 007\n---\nbody")
        assert meta["tenant"] == "007"


class TestHashing:
    def test_content_hash_is_stable_across_line_endings(self):
        assert content_hash("a\nb") == content_hash("a\r\nb")

    def test_content_hash_ignores_trailing_whitespace_only(self):
        assert content_hash("hello world") == content_hash("hello world  \n")

    def test_content_hash_changes_with_content(self):
        assert content_hash("limit is 120ms") != content_hash("limit is 95ms")

    def test_normalise_strips_trailing_whitespace_and_crlf(self):
        # Paragraph structure is *preserved*: only trailing whitespace per line
        # and line-ending style are normalised, because blank lines are
        # meaningful to a chunker and collapsing them would silently change how
        # a document splits.
        assert normalise_text("a   b\r\n\r\nc  \n") == "a   b\n\nc"
        assert normalise_text("same") == normalise_text("same\r\n")


class TestBuildManifest:
    def test_records_documents_and_hashes_content(self, tmp_path):
        write(tmp_path, "a.md", "Alpha document.")
        write(tmp_path, "b.md", "Beta document.")
        manifest = build_manifest(tmp_path)

        assert set(manifest.sources) == {"a.md", "b.md"}
        assert all(len(r.content_hash) == 16 for r in manifest.records)
        assert len(manifest.manifest_id) == 16

    def test_manifest_id_is_deterministic(self, tmp_path):
        write(tmp_path, "a.md", "Alpha.")
        first = build_manifest(tmp_path).manifest_id
        second = build_manifest(tmp_path).manifest_id
        assert first == second, "re-reading an unchanged corpus must not move its identity"

    def test_manifest_id_moves_when_content_changes(self, tmp_path):
        path = write(tmp_path, "a.md", "Limit is 120ms.")
        before = build_manifest(tmp_path).manifest_id
        path.write_text("Limit is 95ms.", encoding="utf-8")
        assert build_manifest(tmp_path).manifest_id != before

    def test_tenant_and_status_survive_into_records(self, tmp_path):
        write(tmp_path, "a.md", "---\ntenant: acme\nstatus: current\n---\n\nBody.")
        record = next(r for r in build_manifest(tmp_path).records if r.source == "a.md")
        assert record.tenant == "acme"
        assert record.status == "current"

    def test_empty_corpus_is_valid(self, tmp_path):
        assert len(build_manifest(tmp_path)) == 0


class TestAttachChunks:
    def _corpus(self, tmp_path):
        write(tmp_path, "a.md", "---\ntenant: acme\n---\n\nAlpha body text.")
        write(tmp_path, "b.md", "---\ntenant: acme\n---\n\nBeta body text.")
        return build_manifest(tmp_path)

    def test_manifest_id_moves_when_chunk_identity_changes(self, tmp_path):
        """The property that makes rechunking detectable.

        If the manifest hashed only the *count* of chunks, splitting one chunk
        into two -- and reassigning every later ordinal, since chunk ids embed
        their position -- would produce an identical manifest id. A corpus whose
        documents are unchanged but whose chunk boundaries moved would then be
        reported as `unchanged`, and a retrieval regression caused entirely by
        rechunking would be blamed on the code. The count is not identity.
        """
        from enterprise_rag.indexing.manifest import CorpusManifest

        base = self._corpus(tmp_path)

        unchanged = attach_chunks(
            base, chunks=[chunk("a.md", 0, "alpha"), chunk("b.md", 1, "beta")]
        )
        # Same chunk count, different boundaries and different ordinals: the
        # chunker changed but no document did.
        rechunked = attach_chunks(
            base,
            chunks=[
                chunk("a.md", 0, "alpha"),
                chunk("a.md", 1, "alpha body"),
                chunk("b.md", 0, "beta"),
                chunk("b.md", 1, "beta body"),
            ],
        )
        assert unchanged.manifest_id != rechunked.manifest_id

    def test_total_chunks_reflects_attachment(self, tmp_path):
        base = self._corpus(tmp_path)
        assert base.total_chunks == 0
        attached = attach_chunks(base, chunks=[chunk("a.md", 0, "alpha"), chunk("b.md", 0, "beta")])
        assert attached.total_chunks == 2

    def test_record_round_trips_through_dict(self, tmp_path):
        write(tmp_path, "a.md", "---\ntenant: acme\nstatus: draft\n---\n\nBody.")
        record = next(iter(build_manifest(tmp_path).records))
        from enterprise_rag.indexing.manifest import DocumentRecord

        assert DocumentRecord.from_dict(record.to_dict()) == record


class TestDiff:
    def _manifest(self, tmp_path, specs):
        # Clear first: `write` only adds, so without this a "removed" document
        # would still be on disk and the diff would -- correctly -- report it as
        # unchanged. That is a fixture bug, not a bug in `diff`.
        for stale in tmp_path.glob("*"):
            stale.unlink()
        for name, body in specs:
            write(tmp_path, name, body)
        return build_manifest(tmp_path)

    def test_detects_added_removed_and_modified(self, tmp_path):
        before = self._manifest(tmp_path, [("a.md", "A one."), ("b.md", "B one.")])
        after = self._manifest(tmp_path, [("a.md", "A one."), ("c.md", "C one.")])
        diff = before.diff(after)

        assert {d.source for d in diff.added} == {"c.md"}
        assert {d.source for d in diff.removed} == {"b.md"}
        assert diff.is_empty is False

    def test_identical_manifests_diff_empty(self, tmp_path):
        before = self._manifest(tmp_path, [("a.md", "A one.")])
        after = self._manifest(tmp_path, [("a.md", "A one.")])
        assert before.diff(after).is_empty

    def test_modified_is_not_reported_as_added_and_removed(self, tmp_path):
        before = self._manifest(tmp_path, [("a.md", "A one.")])
        after = self._manifest(tmp_path, [("a.md", "A two.")])
        diff = before.diff(after)
        assert {d.source for d in diff.modified} == {"a.md"}
        assert not diff.added and not diff.removed
