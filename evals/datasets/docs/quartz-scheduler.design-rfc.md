---
version: 0.1
updated: 2025-01-05
owner: team-gateway
status: draft
tenant: public
---
# quartz-scheduler design-rfc (draft, not approved)

This draft discusses p99 request latency for quartz-scheduler, the error budget, and the
queue depth threshold. It is an unapproved working document and the numbers in
it are proposals, not commitments.

## Latency proposal

A p99 request latency of 800 milliseconds is proposed for quartz-scheduler. This
proposal predates the gateway rewrite and was never adopted.

## Error budget proposal

An error budget of 1.0 percent is discussed but not ratified.
