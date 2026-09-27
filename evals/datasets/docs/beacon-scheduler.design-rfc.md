---
version: 0.1
updated: 2025-01-05
owner: team-scheduler
status: draft
tenant: public
---
# beacon-scheduler design-rfc (draft, not approved)

This draft discusses p99 request latency for beacon-scheduler, the error budget, and the
queue depth threshold. It is an unapproved working document and the numbers in
it are proposals, not commitments.

## Latency proposal

A p99 request latency of 650 milliseconds is proposed for beacon-scheduler. This
proposal predates the scheduler rewrite and was never adopted.

## Error budget proposal

An error budget of 0.5 percent is discussed but not ratified.
