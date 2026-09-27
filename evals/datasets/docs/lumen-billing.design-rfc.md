---
version: 0.1
updated: 2025-01-05
owner: team-ingest
status: draft
tenant: public
---
# lumen-billing design-rfc (draft, not approved)

This draft discusses p99 request latency for lumen-billing, the error budget, and the
queue depth threshold. It is an unapproved working document and the numbers in
it are proposals, not commitments.

## Latency proposal

A p99 request latency of 1050 milliseconds is proposed for lumen-billing. This
proposal predates the ingest rewrite and was never adopted.

## Error budget proposal

An error budget of 1.0 percent is discussed but not ratified.
