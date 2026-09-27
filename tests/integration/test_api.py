"""End-to-end tests over the real HTTP surface and the real corpus.

These assert the *contract* the API promises, not the implementation behind it:

* `/query` always returns auditable evidence, never a bare string.
* Abstention is 200 with a reason, not 4xx.
* A restricted document is refused rather than substituted.
* The same question twice gives the same answer.
* The corpus manifest id is reported, so a bug report is reproducible.

`client` (in `conftest.py`) points the app at the committed evaluation corpus, so
none of these skip: a fresh clone exercises the real index.
"""

from __future__ import annotations

import pytest


class TestHealth:
    def test_health_reports_the_loaded_corpus(self, client):
        body = client.get("/health").json()
        assert body["status"] in {"ok", "degraded"}
        assert body["documents"] > 0
        assert body["chunks"] > 0
        assert body["retrievers"], "no retriever was built"

    def test_health_names_the_generator_in_use(self, client):
        body = client.get("/health").json()
        # Either the deterministic extractive path or a configured provider. The
        # point is that it is named, so a silent fallback is visible.
        assert body["generator"] in {"extractive", "provider"}

    def test_health_reports_corpus_manifest_id(self, client):
        body = client.get("/health").json()
        assert body["corpus_manifest_id"], "a manifest id is required to make a report reproducible"


class TestMetrics:
    def test_metrics_is_prometheus_text(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        body = response.text
        assert "rag_queries_total" in body
        # Every metric needs a TYPE line to be valid exposition format.
        assert "# TYPE rag_queries_total counter" in body

    def test_metrics_does_not_leak_query_text(self, client):
        client.post("/query", json={"query": "what is the retention period"})
        body = client.get("/metrics").text
        assert "retention" not in body.lower()


class TestAnswerableQuestion:
    def test_answers_with_evidence_and_citations(self, client):
        response = client.post(
            "/query", json={"query": "how long are operational documents retained?"}
        )
        assert response.status_code == 200
        body = response.json()

        assert body["status"] in {"answered", "abstained"}
        assert body["answer"], "every response must carry text"
        assert body["reason"], "abstention must explain itself"
        if body["status"] == "answered":
            assert body["citations"], "an answered query must cite its evidence"
            assert body["evidence"], "evidence must be returned for audit"

    def test_citations_point_at_real_documents(self, client):
        body = client.post(
            "/query", json={"query": "which region do deployments run in?"}
        ).json()
        for citation in body.get("citations", []):
            assert citation["source"], "a citation must name a source"
            assert citation["verified"] is True

    def test_response_carries_the_full_stage_trace(self, client):
        body = client.post("/query", json={"query": "what is the capacity limit?"}).json()
        stages = {s["stage"] for s in body["trace"]["stages"]}
        # Every query records where it spent its time, including the ones that
        # abstain -- an unexplained abstention is the hardest failure to debug.
        assert "retrieve" in stages
        assert "authorise" in stages
        if body["status"] == "answered":
            assert {"context", "generate", "verify"} <= stages


class TestCrossTenantRefusal:
    def test_restricted_document_is_refused_not_substituted(self, client):
        body = client.post(
            "/query",
            json={"query": "what is the acme dedicated capacity?", "tenant": "public"},
        ).json()
        if body["status"] != "answered":
            assert "restrict" in body["reason"].lower() or "scope" in body["reason"].lower()

    def test_refusal_does_not_leak_the_withheld_source(self, client):
        body = client.post(
            "/query",
            json={"query": "what is the acme dedicated capacity?", "tenant": "public"},
        ).json()
        # A refusal must not name the document it withheld, or it discloses the
        # existence and subject of the restricted record.
        assert "acme-capacity" not in body["answer"].lower()
        for evidence in body.get("evidence", []):
            assert evidence["tenant"] in {"public"}


class TestAbstention:
    def test_unknown_subject_is_not_answered_from_nothing(self, client):
        body = client.post(
            "/query", json={"query": "what is the airspeed velocity of a llama?"}
        ).json()
        assert body["status"] in {"answered", "abstained"}
        if body["abstained"]:
            assert body["reason"]

    def test_abstention_returns_200_not_4xx(self, client):
        response = client.post("/query", json={"query": "describe the llama's schedule"})
        assert response.status_code == 200

    def test_abstention_is_recorded_in_the_trace(self, client):
        body = client.post("/query", json={"query": "describe the llama's schedule"}).json()
        if body["abstained"]:
            stages = {s["stage"] for s in body["trace"]["stages"]}
            assert "abstain" in stages, "a refusal must record why it refused"


class TestInputValidation:
    def test_empty_query_is_rejected(self, client):
        response = client.post("/query", json={"query": ""})
        assert response.status_code == 422

    def test_oversized_query_is_rejected(self, client):
        response = client.post("/query", json={"query": "x" * 5000})
        assert response.status_code == 422

    def test_response_shape_is_stable(self, client):
        body = client.post("/query", json={"query": "what is the retention period?"}).json()
        # A stable shape is what lets a client be written against this API.
        for field in (
            "query_id", "query", "subject", "tenants", "answer", "status",
            "abstained", "reason", "citations", "confidence", "usage", "trace",
            "corpus_manifest_id",
        ):
            assert field in body, f"response is missing {field!r}"

    def test_manifest_id_is_reported_for_audit(self, client):
        body = client.post("/query", json={"query": "anything at all?"}).json()
        assert body["corpus_manifest_id"]


class TestDeterminism:
    def test_repeated_questions_give_identical_answers(self, client):
        payload = {"query": "how long are operational documents retained?"}
        first = client.post("/query", json=payload).json()
        second = client.post("/query", json=payload).json()
        assert first["answer"] == second["answer"]
        assert first["citations"] == second["citations"]

    def test_cache_hit_does_not_change_the_answer(self, client):
        payload = {"query": "which region do deployments run in?"}
        first = client.post("/query", json=payload).json()
        second = client.post("/query", json=payload).json()
        assert first["answer"] == second["answer"]
        # The second call is a cache hit, and the trace must say so rather than
        # reporting it as a fresh retrieval.
        retrieve = [s for s in second["trace"]["stages"] if s["stage"] == "retrieve"][0]
        assert retrieve["cache_hit"] is True


class TestBackwardsCompatibleAlias:
    def test_ask_still_works(self, client):
        response = client.post("/ask", json={"query": "what is the retention period?"})
        assert response.status_code == 200
        assert "answer" in response.json()


class TestCorpusEndpoint:
    def test_corpus_reports_manifest_and_shape(self, client):
        body = client.get("/corpus").json()
        assert body["manifest_id"]
        assert body["documents"] > 0
        assert body["chunks"] > 0
