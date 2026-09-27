"""Request and response models for the HTTP surface.

Two conventions worth stating, because both were learned the hard way:

**Abstention is 200.** `status: "abstained"` with a `reason`, never a 4xx. A
4xx makes "I don't know" indistinguishable from "your request was malformed",
and callers retry errors.

**Every response carries its evidence.** The retrieved spans, the stage timings,
the authorization decision and the token/cost accounting all come back. An
answer you cannot check without server access is a claim, not a result.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = [
    "HealthResponse",
    "IndexRequest",
    "IngestRequest",
    "IngestResponse",
    "QueryRequest",
    "QueryResponse",
]


class QueryRequest(BaseModel):
    """A question plus the caller's asserted scope.

    `tenant` and `subject` describe who is asking. See the security note on
    `RagPipeline.query`: a client-supplied tenant is a demonstration of the
    *policy*, not an authentication control. A real deployment must derive the
    principal from verified token claims, because a client-supplied tenant string
    is a client-supplied permission.
    """

    query: str = Field(..., min_length=1, max_length=4000, description="The question.")
    tenant: str = Field("public", description="Tenant scope the caller may read.")
    subject: str = Field("public", description="Authenticated subject identifier.")
    final_k: int | None = Field(None, ge=1, le=50, description="Override result depth.")
    include_evidence: bool = Field(True, description="Return retrieved spans in the response.")


class QueryResponse(BaseModel):
    query_id: str
    query: str
    subject: str
    tenants: list[str]
    answer: str
    status: str = Field(..., description="answered | abstained | refused")
    abstained: bool
    reason: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    confidence: float
    usage: dict[str, Any] = Field(default_factory=dict)
    generator: str = ""
    generation_degraded: bool = False
    withheld_count: int = 0
    injection_findings: int = 0
    degradations: list[str] = Field(default_factory=list)
    context: dict[str, Any] | None = None
    citations_report: dict[str, Any] | None = None
    trace: dict[str, Any] = Field(default_factory=dict)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    corpus_manifest_id: str = ""


class IngestRequest(BaseModel):
    source_dir: str = Field(..., description="Directory of source documents to parse.")
    output_dir: str | None = Field(
        None, description="Where to write parsed Markdown. Defaults to evals/datasets/<corpus_name>."
    )
    corpus_name: str = Field("corpus", description="Name for the output corpus.")
    tenant: str = Field("public", description="Tenant to attribute every document to.")
    status: str = Field("current", description="Lifecycle status for every document.")


class IngestResponse(BaseModel):
    corpus_name: str
    output_dir: str
    documents: int
    manifest_id: str
    tenant: str
    status: str
    written: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class IndexRequest(BaseModel):
    corpus_dir: str | None = Field(None, description="Corpus to index. Defaults to the configured one.")
    rebuild: bool = Field(False, description="Rebuild even if a cached index exists.")


class HealthResponse(BaseModel):
    status: str = Field(..., description="ok | degraded")
    corpus_manifest_id: str
    documents: int
    chunks: int
    retrievers: list[str]
    strategy: str
    generator: str = ""
    capabilities: dict[str, str] = Field(default_factory=dict)
    config_warnings: list[str] = Field(default_factory=list)
    degradations: list[str] = Field(default_factory=list)


class MetricsResponse(BaseModel):
    """Process counters, in Prometheus text exposition format.

    `body` is the scrape payload. It is a string because the exposition format
    is a wire format, not JSON, and a JSON object here would be a second,
    subtly different metrics schema that scrapers cannot read.
    """

    content_type: str = "text/plain; version=0.0.4; charset=utf-8"
    body: str
