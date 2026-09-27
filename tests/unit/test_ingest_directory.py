"""Ingestion from a directory of files.

`ingest_directory` had no test and raised `AttributeError: 'CorpusManifest'
object has no attribute 'documents'` on every call -- it read a `documents`
attribute where the field is `records`. Nothing caught it because nothing ran
it: the pipeline tests exercise `ingest_file` one document at a time, and the
committed evaluation corpus is loaded straight into memory rather than through
the ingest path.

The point of these tests is therefore narrow and deliberate: call the
directory-level function for real, on a real temporary directory, and assert on
the whole return value rather than on "it did not raise". A test that only
checks for the absence of an exception is what let a function that could only
raise pass review for this long.
"""

from __future__ import annotations

from pathlib import Path

from enterprise_rag.ingestion.pipeline import ingest_directory


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_ingest_directory_returns_counts_and_a_manifest_id(tmp_path: Path) -> None:
    source = tmp_path / "in"
    _write(source / "alpha.md", "# Alpha\n\nalpha beta gamma delta epsilon\n")
    _write(source / "beta.txt", "beta gamma delta epsilon zeta\n")
    out = tmp_path / "out"

    result = ingest_directory(source, out)

    assert result["written"] == ["alpha.md", "beta.txt"]
    assert result["skipped"] == []
    assert result["documents"] == 2
    assert result["manifest_id"], "a manifest id is what makes the corpus reproducible"


def test_ingest_directory_reports_what_it_wrote_and_what_it_skipped(tmp_path: Path) -> None:
    source = tmp_path / "in"
    _write(source / "kept.md", "kept content\n")
    # Supported extension, but no text to extract, so nothing is written.
    _write(source / "empty.md", "")
    # An unsupported extension and a nested directory are filtered out before the
    # skip list is built, so they appear in neither `written` nor `skipped`.
    # Asserted here so the behaviour is pinned rather than assumed -- and it is
    # arguably wrong. A file nobody asked for is not a problem, but a file
    # somebody put in the directory expecting it to be indexed and then finding
    # no trace of it is. Reporting every path examined, and why each was
    # excluded, is the honest version of this.
    _write(source / "image.png", "not text")
    (source / "nested").mkdir()
    _write(source / "nested" / "deep.md", "deep content")
    out = tmp_path / "out"

    result = ingest_directory(source, out)

    assert result["written"] == ["kept.md"]
    assert result["skipped"] == ["empty.md"]
    assert result["documents"] == 1


def test_ingest_directory_is_idempotent_on_its_own_output(tmp_path: Path) -> None:
    """Re-ingesting the same files must produce the same manifest id.

    This is the property the whole evaluation rests on: a number measured
    against corpus X has to still mean the same thing later. If a second run
    changed the id, every published result would be unfalsifiable.
    """
    source = tmp_path / "in"
    _write(source / "one.md", "one two three four five\n")
    _write(source / "two.md", "two three four five six\n")

    first = ingest_directory(source, tmp_path / "run1")
    second = ingest_directory(source, tmp_path / "run2")
    third = ingest_directory(source, tmp_path / "run2")

    assert first["manifest_id"] == second["manifest_id"]
    assert second["manifest_id"] == third["manifest_id"]


def test_ingest_directory_creates_a_missing_output_directory(tmp_path: Path) -> None:
    source = tmp_path / "in"
    _write(source / "only.md", "content here\n")
    out = tmp_path / "does" / "not" / "exist" / "yet"

    result = ingest_directory(source, out)

    assert out.is_dir()
    assert result["documents"] == 1
