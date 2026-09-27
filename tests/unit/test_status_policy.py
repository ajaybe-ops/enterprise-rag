"""Lifecycle status policy: exclusion, demotion, and `partition`.

`StatusPolicy.partition` had a bug that no test could catch, because nothing
called it. It passed the whole `metadata` mapping to `is_demoted`, whose
signature takes a status *string* and immediately does `(status or "").strip()`.
Given a non-empty dict that raises `AttributeError: 'dict' object has no
attribute 'strip'`, so the method crashed on the first chunk that was not
superseded. It survived review because it is unreachable: the pipeline applies
status through `_apply_status` and `multiplier` instead, and `partition` is a
convenience wrapper nobody wired up.

An untested, uncalled, broken function is still broken, and the fix is to test
it -- or delete it. It is tested here because it is the readable expression of
the policy, and a policy that is only expressed in two harder-to-read places is
a policy that will drift between them.
"""

from __future__ import annotations

import pytest

from enterprise_rag.retrieval.status import StatusPolicy
from enterprise_rag.types import Chunk


def _chunk(source: str, ordinal: int = 0, **metadata: str) -> Chunk:
    return Chunk(
        chunk_id=f"{source}:{ordinal}",
        source=source,
        ordinal=ordinal,
        text="body",
        metadata=metadata,
    )


@pytest.fixture
def policy() -> StatusPolicy:
    return StatusPolicy()


def test_partition_splits_by_status_without_raising(policy: StatusPolicy) -> None:
    """The regression: three different statuses, one call, no exception.

    Under the old code this raised on the `current` chunk, because it reached
    `is_demoted` with a dict.
    """
    kept = _chunk("current.md", status="current")
    demoted = _chunk("draft.md", status="draft")
    excluded = _chunk("old.md", status="superseded")

    k, d, e = policy.partition([kept, demoted, excluded])

    assert k == [kept]
    assert d == [demoted]
    assert e == [excluded]


def test_partition_leaves_an_unclassified_chunk_kept(policy: StatusPolicy) -> None:
    """No status means unclassified, which means untouched.

    Not demoted and not excluded. Defaulting an absent status to `draft` would
    quietly demote most of a corpus that simply has no front matter.
    """
    chunk = _chunk("plain.md")

    k, d, e = policy.partition([chunk])

    assert k == [chunk]
    assert d == []
    assert e == []


def test_partition_preserves_order_within_each_bucket(policy: StatusPolicy) -> None:
    """Order is part of the contract: these lists are handed to a ranker."""
    chunks = [
        _chunk("a.md", 0, status="current"),
        _chunk("b.md", 1, status="draft"),
        _chunk("c.md", 2, status="current"),
        _chunk("d.md", 3, status="draft"),
    ]

    k, d, e = policy.partition(chunks)

    assert [c.source for c in k] == ["a.md", "c.md"]
    assert [c.source for c in d] == ["b.md", "d.md"]
    assert e == []


def test_partition_excludes_before_demoting(policy: StatusPolicy) -> None:
    """A superseded draft is excluded, not demoted.

    Exclusion is the stronger rule and must win. If demotion were checked first
    a superseded document would keep competing at a reduced score, which is the
    behaviour exclusion exists to prevent.
    """
    chunk = _chunk("gone.md", status="superseded")

    # Both predicates agree here, so assert the precedence directly.
    assert policy.is_excluded(chunk.metadata) is True
    assert policy.is_demoted("draft") is True

    _k, d, e = policy.partition([chunk])
    assert e == [chunk]
    assert d == []


def test_is_excluded_and_is_demoted_take_different_arguments(policy: StatusPolicy) -> None:
    """Pin the asymmetry that caused the bug.

    `is_excluded` reads a mapping; `is_demoted` reads a status string. Passing
    one to the other is the mistake this module made, and it is invisible in
    review because the names are similar and both are "about status".
    """
    assert policy.is_excluded({"status": "superseded"}) is True
    assert policy.is_excluded({}) is False

    # A dict where a string is expected is what `partition` used to do.
    with pytest.raises((AttributeError, TypeError)):
        policy.is_demoted({"status": "draft"})  # type: ignore[arg-type]


def test_status_matching_is_case_and_whitespace_insensitive(policy: StatusPolicy) -> None:
    """Front matter is written by people, so ` Draft ` must behave as `draft`."""
    assert policy.is_demoted(" Draft ") is True
    assert policy.is_demoted("DRAFT") is True
    assert policy.is_excluded({"status": " Superseded "}) is True
