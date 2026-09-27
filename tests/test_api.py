"""End-to-end API tests against the generated corpus.

Runs against `evaluation/generated/`, which is reproduced byte-identically by
`python scripts/generate_corpus.py`. If the corpus is absent the module skips
rather than failing: an absent build product is a missing `make generate`, not a
broken service, and a test suite that reports a bug for an unrun build step
teaches people to ignore it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "evaluation" / "generated" / "docs"
pytestmark = pytest.mark.skipif(
    not CORPUS.is_dir(), reason="run: python scripts/generate_corpus.py"
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


class TestHealth:
    def test_health_reports_the_loaded_corpus(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["documents"] > 0
        assert body["chunks"] > 0
        assert body["degradations"] == []

    def test_health_exposes_which_retrievers_ran(self, client):
        # Without this, a dense retriever silently falling back to LSA looks
        # identical to a healthy system. A degradation you cannot see is a
        # benchmark you cannot trust.
        assert set(client.get("/health").json()["retrievers"]) >= {"sparse", "dense"}


class TestAnswerableQuestion:
    def test_answers_with_citations(self, client):
        body = client.post(
            "/ask", json={"query": "What is the p99 request latency objective for atlas-ingest?", "tenant": "public"}
        ).json()
        assert body["abstained"] is False
        assert body["citations"], "an answered question with no citation is an unsupported claim"
        assert all("source" in c for c in body["citations"])

    def test_citations_point_at_real_sources(self, client):
        body = client.post(
            "/ask", json={"query": "What is the p99 request latency objective for atlas-ingest?", "tenant": "public"}
        ).json()
        known = {p.name for p in CORPUS.glob("*.md")}
        assert {c["source"] for c in body["citations"]} <= known


class TestCrossTenantRefusal:
    def test_restricted_capacity_is_refused_for_public_caller(self, client):
        body = client.post(
            "/ask", json={"query": "What is the contracted capacity for atlas-ingest?", "tenant": "public"}
        ).json()
        assert body["abstained"] is True
        assert "tenant" in body["reason"].lower()

    def test_refusal_does_not_leak_the_withheld_source_name_in_the_answer(self, client):
        body = client.post(
            "/ask", json={"query": "What is the contracted capacity for atlas-ingest?", "tenant": "public"}
        ).json()
        assert "capacity" not in body["answer"].lower()
        assert "restricted" not in body["answer"].lower()


class TestAbstention:
    def test_unknown_subject_abstains(self, client):
        body = client.post(
            "/ask", json={"query": "What is the p99 latency objective for zephyr-dispatcher?", "tenant": "public"}
        ).json()
        assert body["abstained"] is True
        assert body["citations"] == []

    def test_abstention_gives_a_reason(self, client):
        # A silent abstention is indistinguishable from a hung request, and gets
        # debugged by a human instead of by a log line.
        body = client.post(
            "/ask", json={"query": "What is the p99 latency objective for zephyr-dispatcher?", "tenant": "public"}
        ).json()
        assert body["reason"]


class TestInputValidation:
    def test_empty_query_is_rejected(self, client):
        assert client.post("/ask", json={"query": "", "tenant": "public"}).status_code == 422

    def test_response_shape_is_stable(self, client):
        body = client.post(
            "/ask", json={"query": "What is the p99 request latency objective for atlas-ingest?", "tenant": "public"}
        ).json()
        # `subject`/`tenants` record the scope the answer was produced under, so a
        # captured response can be audited without re-deriving the caller's
        # identity from the request.
        for field in (
            "answer", "abstained", "citations", "evidence", "confidence", "reason",
            "subject", "tenants", "withheld_count", "corpus_manifest_id",
        ):
            assert field in body

    def test_manifest_id_is_reported_for_audit(self, client):
        # An answer with no recorded corpus identity cannot be reproduced later,
        # which is what makes a quality complaint debuggable at all.
        body = client.post(
            "/ask", json={"query": "What is the p99 request latency objective for atlas-ingest?", "tenant": "public"}
        ).json()
        assert body["corpus_manifest_id"]


class TestDeterminism:
    def test_repeated_questions_give_identical_answers(self, client):
        payload = {"query": "What is the p99 request latency objective for atlas-ingest?", "tenant": "public"}
        first, second = client.post("/ask", json=payload).json(), client.post("/ask", json=payload).json()
        assert first["answer"] == second["answer"]
        assert [c["source"] for c in first["citations"]] == [c["source"] for c in second["citations"]]
        # A retrieval service that returns different evidence for the same query
        # cannot be regression-tested at all, since the baseline is not stable.
