---
version: 0.1
updated: 2025-01-05
owner: team-replicator
status: draft
tenant: public
---
# cinder-replicator faq (draft, not approved)

This draft discusses p99 request latency for cinder-replicator, the error budget, and the
queue depth threshold. It is an unapproved working document and the numbers in
it are proposals, not commitments.

## Latency proposal

A p99 request latency of 1250 milliseconds is proposed for cinder-replicator. This
proposal predates the replicator rewrite and was never adopted.

## Error budget proposal

An error budget of 0.5 percent is discussed but not ratified.
