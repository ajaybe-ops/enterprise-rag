---
version: 0.1
updated: 2025-01-05
owner: team-scheduler
status: draft
tenant: public
---
# nimbus-registry glossary (draft, not approved)

This draft discusses p99 request latency for nimbus-registry, the error budget, and the
queue depth threshold. It is an unapproved working document and the numbers in
it are proposals, not commitments.

## Latency proposal

A p99 request latency of 520 milliseconds is proposed for nimbus-registry. This
proposal predates the scheduler rewrite and was never adopted.

## Error budget proposal

An error budget of 1.0 percent is discussed but not ratified.
