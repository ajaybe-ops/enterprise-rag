"""Compare a committed corpus manifest against a freshly regenerated one.

This is a committed script rather than an inline ``python -c`` in the workflow
because that check has to produce the same result on Ubuntu and Windows, and
nested quoting across bash, PowerShell and Python is where checks like this
quietly break on exactly one of the two.

What it asserts
---------------
The committed manifest must exist -- an evaluation with no pinned dataset is not
reproducible, it is merely repeatable -- and its ``manifest_id`` must equal the
freshly generated one. The id is content-addressed over the chunk identities, so
equality means the corpus is byte-for-byte reproducible, not merely the same
number of documents.

A mismatch is reported as a diff rather than a bare assertion failure, because
"manifest drifted" on its own tells whoever hits it nothing about what changed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


def load_manifest(path: pathlib.Path) -> dict[str, Any]:
    """Read a manifest, failing with the path rather than a traceback."""
    if not path.is_file():
        raise SystemExit(
            f"manifest not found: {path}\n"
            "The evaluation dataset must be committed for results to be reproducible."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc}") from exc
    if "manifest_id" not in payload:
        raise SystemExit(f"{path} has no 'manifest_id' key; refusing to compare")
    return payload


def describe_drift(committed: dict[str, Any], fresh: dict[str, Any]) -> list[str]:
    """Human-readable differences between two manifests, best effort.

    The document sets are compared as well as the id, because knowing *which*
    documents moved is the difference between a five-second and a five-minute
    investigation. Any structural difference in shape is reported rather than
    assumed.
    """
    notes: list[str] = []
    # The manifest stores documents under `records`; `documents` is the key used
    # in a manifest *diff*, not in a manifest file. Reading the wrong one yields
    # an empty set, which makes the document diff silently report "no change" on
    # exactly the run it exists to explain.
    committed_docs = {r.get("source") for r in committed.get("records", ())}
    fresh_docs = {r.get("source") for r in fresh.get("records", ())}
    for label, difference in (
        ("added", fresh_docs - committed_docs),
        ("removed", committed_docs - fresh_docs),
    ):
        if difference:
            sample = ", ".join(sorted(str(x) for x in difference)[:5])
            more = "" if len(difference) <= 5 else f" (+{len(difference) - 5} more)"
            notes.append(f"{len(difference)} document(s) {label}: {sample}{more}")
    for key in ("dataset_id", "chunk_count", "document_count", "chunk_size"):
        before, after = committed.get(key), fresh.get(key)
        if before != after:
            notes.append(f"{key}: {before!r} -> {after!r}")
    return notes


def main(argv: list[str] | None = None) -> int:
    # RawDescriptionHelpFormatter so the module docstring -- which is the
    # usage documentation -- stays readable instead of being reflowed.
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--committed", required=True, type=pathlib.Path, help="the manifest in git")
    parser.add_argument("--regenerated", required=True, type=pathlib.Path, help="one just generated")
    args = parser.parse_args(argv)

    committed = load_manifest(args.committed)
    fresh = load_manifest(args.regenerated)

    if committed["manifest_id"] == fresh["manifest_id"]:
        print(f"manifest {fresh['manifest_id']} reproducible")
        return 0

    print("corpus manifest drifted", file=sys.stderr)
    print(f"  committed   {committed['manifest_id']}", file=sys.stderr)
    print(f"  regenerated {fresh['manifest_id']}", file=sys.stderr)
    for note in describe_drift(committed, fresh):
        print(f"  {note}", file=sys.stderr)
    print(
        "\nIf this change is intended, regenerate and commit the dataset:\n"
        "  python scripts/generate_corpus.py --out evals/datasets --clean",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
