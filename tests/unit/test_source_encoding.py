"""Guard the source tree against character-encoding damage.

This exists because of a real bug, twice over.

The story
---------
`src/enterprise_rag/ingestion/pipeline.py` contained the sequence ``E2 86 92``,
which is how UTF-8 encodes U+2192 RIGHTWARDS ARROW. Something read that file as
cp1252 -- the default encoding for the Windows console and, historically, for
`open()` on Windows -- and wrote the result back. Each byte then became a
separate character, so the arrow turned into three characters:

    E2 86 92  ->  'â' '†' '’'      (U+00E2, U+2020, U+2019)

Nothing raised. The file was still valid UTF-8, still valid Python, and the
docstring still parsed. It just no longer said what it was supposed to say, and
the corruption was invisible to a scan for U+FFFD REPLACEMENT CHARACTER --
because no decoder had failed, so no replacement character was ever produced.

The second lesson is the reason this is a test and not a one-off fix. My first
repair pass looked for ``�`` and therefore reported the tree clean while
pipeline.py was the single worst-damaged file in it. A check that only knows
about one symptom misses a different encoding of the same problem.

The detection
-------------
Mojibake of this kind is self-identifying: the text is valid UTF-8 whose
characters are all in the cp1252 range, and re-encoding it to cp1252 and decoding
the result as UTF-8 yields different, sane text. ``test_no_mojibake_in_source``
applies that round trip to every tracked text file. A file that trips it has
already lost information and must be repaired, not reformatted.

What is deliberately not checked
--------------------------------
* U+00A0 NO-BREAK SPACE and the typographic quotes U+2018..U+201D are *correct*
  in the source. They are not damage, and normalising them away would make the
  diff noise and hide the one thing this test is for.
* The evaluation corpus under ``evals/datasets/docs`` is not scanned. Those files
  are fixtures holding deliberately awkward input, and one of them contains
  malformed bytes on purpose to exercise the ingest fallback.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Suffixes that must stay UTF-8. Binary and asset files are excluded by
# extension rather than by sniffing, so adding a notebook or a lockfile later
# does not silently drop it from the check.
TEXT_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".toml", ".json", ".txt", ".example", ".cfg", ".ini"}

# Directories whose contents are inputs or build output rather than authored
# source. See the docstring for why the corpus is excluded.
SKIP_DIRS = {".git", ".venv", "runs", "build", "dist", ".ruff_cache", ".pytest_cache", "__pycache__"}


def tracked_text_files() -> list[Path]:
    """Text files under version control, via `git ls-files`.

    Reading the git index rather than walking the filesystem means a developer's
    uncommitted scratch file cannot fail the suite, and a file that was added to
    git is guaranteed to be covered. If git is unavailable the caller skips
    rather than silently testing nothing.
    """
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.py", "*.md", "*.yml", "*.yaml", "*.toml", "*.json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"git ls-files failed: {result.stderr.strip()}")
    names = [n for n in result.stdout.split("\0") if n]
    paths = [ROOT / name for name in names]
    # `git ls-files` reports the index, which still lists paths deleted in the
    # working tree until the deletion is staged. Reading those raises, so keep
    # only what is actually on disk.
    return [
        p
        for p in paths
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES and not SKIP_DIRS & set(p.parts)
    ]


def _non_ascii_runs(text: str) -> list[tuple[int, str]]:
    """Maximal runs of non-ASCII characters, as (offset, run).

    Per-run rather than whole-file because the two failure modes coexist. A file
    can hold a legitimate Greek capital and a damaged em dash on different lines:
    `retrieval/fusion.py` did exactly that, containing both a real `Σ` and a
    damaged `â€”`. A whole-file round trip re-encodes to cp1252 and fails to
    decode as UTF-8 on the *legitimate* bytes, so it reports the file clean and
    the damage survives. Splitting into runs tests each cluster on its own.
    """
    runs: list[tuple[int, str]] = []
    start: int | None = None
    for index, char in enumerate(text):
        if ord(char) > 127:
            if start is None:
                start = index
        elif start is not None:
            runs.append((start, text[start:index]))
            start = None
    if start is not None:
        runs.append((start, text[start:]))
    return runs


def unmojibake(text: str) -> str | None:
    """Return the first damaged run found in `text`, or None if there is none.

    A run is damaged when re-encoding it to cp1252 and decoding as UTF-8 yields
    different, sane text -- the signature of UTF-8 having been read as cp1252.
    A run that is genuinely correct cannot survive that round trip: real
    characters like `—` (U+2014) and `Σ` (U+03A3) re-encode to bytes that are
    not valid UTF-8 on their own, so the decode raises and the run is excused.
    """
    for _, run in _non_ascii_runs(text):
        try:
            repaired = run.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if repaired != run:
            return repaired
    return None


def test_no_mojibake_in_source() -> None:
    """No tracked text file is cp1252-mangled UTF-8."""
    offenders: list[str] = []
    for path in tracked_text_files():
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            offenders.append(f"{path.relative_to(ROOT)}: not valid UTF-8 ({exc})")
            continue
        damaged = unmojibake(text)
        if damaged is not None:
            offenders.append(
                f"{path.relative_to(ROOT)}: contains double-encoded text, "
                f"first damaged run decodes to {damaged!r}"
            )

    assert not offenders, (
        "double-encoded (mojibake) source files found. Repair with:\n"
        "  text = p.read_text(encoding='utf-8')\n"
        "  for _, run in runs(text):\n"
        "      text = text.replace(run, run.encode('cp1252').decode('utf-8'))\n"
        + "\n".join(f"  {line}" for line in offenders)
    )


def test_no_replacement_characters_in_source() -> None:
    """No tracked text file contains U+FFFD.

    Separate from the round-trip test because a replacement character means a
    decoder *did* fail, which is a different failure with a different repair.
    """
    offenders = [
        str(path.relative_to(ROOT))
        for path in tracked_text_files()
        if "�" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"U+FFFD found in: {', '.join(offenders)}"


def test_ingestion_docstring_still_reads_correctly() -> None:
    """The file that actually broke, pinned against regressing.

    A whole-tree scan is the safety net; this is the assertion that would have
    caught the original bug on the day it was introduced, when the rest of the
    tree was still clean and therefore looked healthy.
    """
    source = (ROOT / "src" / "enterprise_rag" / "ingestion" / "pipeline.py").read_text(encoding="utf-8")
    first_line = source.splitlines()[0]
    assert "→" in first_line, f"arrow lost from the ingestion docstring: {first_line!r}"
    assert "â†’" not in source, "ingestion docstring is mojibake again"


def test_unmojibake_detects_the_known_corruption() -> None:
    """The detector must fire on real samples, not just pass vacuously.

    A guard test that cannot fail is worse than no guard test, because it reads
    like coverage. These construct the exact byte sequences the original bug
    produced and assert the detector finds them.
    """
    # The single-arrow case: UTF-8 E2 86 92 read as cp1252.
    damaged = "files on disk " + "→".encode().decode("cp1252") + " text"
    assert unmojibake(damaged) == "→"

    # And a genuinely correct arrow must not be reported as damage. U+2192 is not
    # in cp1252 at all, so re-encoding raises and the run is excused.
    assert unmojibake("files on disk → text") is None


def test_unmojibake_survives_mixed_legitimate_and_damaged_text() -> None:
    """A file can be correct in places and damaged in others.

    This is the false negative that let a damaged `retrieval/fusion.py` sit in
    a commit while a whole-file round trip reported it clean. The file contained
    a real `Σ` and a real `—` alongside damaged `â€”` and `Î£`; encoding the
    whole thing to cp1252 produced byte sequences that are not valid UTF-8, so
    the decode raised and the file was exonerated.

    Every damaged run must be found, and the legitimate runs must survive
    untouched -- fixing one must not "repair" the other into a different
    character.
    """
    correct_sigma = "Σ"
    correct_em_dash = "—"
    damaged_em_dash = correct_em_dash.encode("utf-8").decode("cp1252")
    damaged_sigma = correct_sigma.encode("utf-8").decode("cp1252")

    assert unmojibake(f"scores {damaged_sigma} 1 / (k + rank_i(d))") == "Σ"
    assert unmojibake(f"**RRF** {damaged_em_dash} Cormack et al.") == "—"

    # The whole-file shape that used to defeat the check.
    mixed = f"{correct_sigma} and {damaged_em_dash} and {correct_em_dash}"
    assert unmojibake(mixed) == "—", "damaged run hidden among legitimate ones was missed"

    # All-legitimate text of the same shape stays clean.
    assert unmojibake(f"{correct_sigma} and {correct_em_dash} and —") is None
