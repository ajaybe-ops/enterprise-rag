"""Shared test fixtures.

The suite is split by what it needs, not by what it is convenient to group:

* ``unit/`` tests one module against synthetic input and needs nothing on disk.
* ``integration/`` builds a real index over a real corpus and a real app.
* ``evaluation/`` asserts on the committed probe set and the quality gate.

Two things are centralised here because getting them wrong silently weakens every
test that depends on them: the ``enterprise_rag`` import path, and resetting
process-global state between tests.

On the reset
------------
`config.settings` and `api.security` both hold process-global state -- the cached
`Settings` singleton and the rate-limit windows. A test that installs a setting
and does not restore it makes every later test depend on execution order, which
is the failure mode that turns a green suite into an untrustworthy one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# The committed evaluation corpus. Present in a fresh clone, so integration tests
# do not skip: `git clone && pytest` must exercise the real thing.
EVAL_CORPUS = ROOT / "evals" / "datasets" / "docs"
EVAL_PROBES = ROOT / "evals" / "datasets" / "probes.json"
# Four documents, two tenants, one superseded. Enough to exercise authorization
# and status policy without making unit tests slow.
FIXTURE_CORPUS = Path(__file__).parent / "fixtures" / "corpus"


@pytest.fixture(autouse=True)
def _reset_process_state():
    """Restore global config and rate-limit state around every test."""
    from enterprise_rag.api.security import reset_rate_limits
    from enterprise_rag.config.settings import get_settings, set_settings

    before = get_settings()
    reset_rate_limits()
    try:
        yield
    finally:
        reset_rate_limits()
        set_settings(before)


@pytest.fixture
def eval_corpus() -> Path:
    if not EVAL_CORPUS.is_dir():
        pytest.skip("evaluation corpus missing; run: python scripts/generate_corpus.py")
    return EVAL_CORPUS


@pytest.fixture
def fixture_corpus() -> Path:
    return FIXTURE_CORPUS


@pytest.fixture
def principal():
    from enterprise_rag.security.access import Principal

    return Principal(subject="analyst", tenants=frozenset({"public"}))


@pytest.fixture
def pipeline(eval_corpus):
    """A real pipeline over the committed corpus, built once per test module."""
    from enterprise_rag.api.deps import build_pipeline
    from enterprise_rag.config.settings import get_settings

    settings = get_settings().with_overrides(corpus_dir=str(eval_corpus))
    built, _degradations = build_pipeline(settings, eval_corpus)
    return built


@pytest.fixture
def client(eval_corpus, monkeypatch):
    """A TestClient whose app is backed by the committed evaluation corpus."""
    from fastapi.testclient import TestClient

    from enterprise_rag.api.app import create_app
    from enterprise_rag.api.deps import state
    from enterprise_rag.config.settings import get_settings, set_settings

    settings = get_settings().with_overrides(
        corpus_dir=str(eval_corpus),
        # Generous, so a test suite does not trip its own limiter; the limiter
        # has its own test.
        rate_limit_per_minute=10_000,
    )
    # The app resolves configuration through `get_settings()` at request time, so
    # the overrides above only take effect once installed. An earlier version of
    # this fixture built `settings` and never installed it: the app silently fell
    # back to the default corpus and the real 20-per-minute limiter, and every
    # API test was passing against a configuration nobody had chosen. Tests
    # asserting on 429s would have been order-dependent for the same reason.
    set_settings(settings)
    state.invalidate()
    with TestClient(create_app()) as test_client:
        yield test_client
    state.invalidate()
