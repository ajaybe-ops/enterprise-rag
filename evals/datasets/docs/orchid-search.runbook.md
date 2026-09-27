---
version: 1.1
updated: 2026-07-16
owner: team-replicator
status: current
tenant: public
---
# orchid-search on-call runbook

Page the on-call engineer for orchid-search when the burn rate exceeds the threshold
or when the service has been in a degraded state for fifteen minutes.

## Alerting

The primary alert fires when error rate stays above the objective for ten
minutes. A secondary alert fires when replica lag exceeds 30 seconds.

## First response

Confirm the alert against the dashboard before taking action. If queue depth is
above 10000, scale the replicator tier first. Do not restart the
deployment to clear a latency alert; the restart is more disruptive than the
symptom.
