---
version: 0.1
updated: 2025-01-05
owner: team-router
status: draft
tenant: public
---
# meridian-notify faq (draft, not approved)

This draft discusses p99 request latency for meridian-notify, the error budget, and the
queue depth threshold. It is an unapproved working document and the numbers in
it are proposals, not commitments.

## Latency proposal

A p99 request latency of 720 milliseconds is proposed for meridian-notify. This
proposal predates the router rewrite and was never adopted.

## Error budget proposal

An error budget of 0.1 percent is discussed but not ratified.
