---
version: 0.1
updated: 2025-01-05
owner: team-compactor
status: draft
tenant: public
---
# pyrite-archive postmortem-draft (draft, not approved)

This draft discusses p99 request latency for pyrite-archive, the error budget, and the
queue depth threshold. It is an unapproved working document and the numbers in
it are proposals, not commitments.

## Latency proposal

A p99 request latency of 580 milliseconds is proposed for pyrite-archive. This
proposal predates the compactor rewrite and was never adopted.

## Error budget proposal

An error budget of 0.5 percent is discussed but not ratified.
