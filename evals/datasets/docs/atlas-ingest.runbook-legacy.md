---
version: 0.1
updated: 2025-01-05
owner: team-ingest
status: draft
tenant: public
---
# atlas-ingest runbook-legacy (draft, not approved)

This draft discusses p99 request latency for atlas-ingest, the error budget, and the
queue depth threshold. It is an unapproved working document and the numbers in
it are proposals, not commitments.

## Latency proposal

A p99 request latency of 520 milliseconds is proposed for atlas-ingest. This
proposal predates the ingest rewrite and was never adopted.

## Error budget proposal

An error budget of 0.1 percent is discussed but not ratified.
