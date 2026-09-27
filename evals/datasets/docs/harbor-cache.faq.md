---
version: 0.1
updated: 2025-01-05
owner: team-scheduler
status: draft
tenant: public
---
# harbor-cache faq (draft, not approved)

This draft discusses p99 request latency for harbor-cache, the error budget, and the
queue depth threshold. It is an unapproved working document and the numbers in
it are proposals, not commitments.

## Latency proposal

A p99 request latency of 800 milliseconds is proposed for harbor-cache. This
proposal predates the scheduler rewrite and was never adopted.

## Error budget proposal

An error budget of 0.5 percent is discussed but not ratified.
