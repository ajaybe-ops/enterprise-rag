---
version: 0.1
updated: 2025-01-05
owner: team-gateway
status: draft
tenant: public
---
# ember-gateway runbook-legacy (draft, not approved)

This draft discusses p99 request latency for ember-gateway, the error budget, and the
queue depth threshold. It is an unapproved working document and the numbers in
it are proposals, not commitments.

## Latency proposal

A p99 request latency of 1000 milliseconds is proposed for ember-gateway. This
proposal predates the gateway rewrite and was never adopted.

## Error budget proposal

An error budget of 0.5 percent is discussed but not ratified.
