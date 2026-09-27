---
version: 2.4
updated: 2026-05-14
owner: team-compactor
status: current
tenant: public
---
# delta-compactor on-call runbook

Page the on-call engineer for delta-compactor when the burn rate exceeds the threshold
or when the service has been in a degraded state for fifteen minutes.

## Alerting

The primary alert fires when error rate stays above the objective for ten
minutes. A secondary alert fires when replica lag exceeds 5 seconds.

## First response

Confirm the alert against the dashboard before taking action. If queue depth is
above 1000, scale the compactor tier first. Do not restart the
deployment to clear a latency alert; the restart is more disruptive than the
symptom.
