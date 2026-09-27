---
version: 3.5
updated: 2026-06-15
owner: team-gateway
status: current
tenant: public
---
# ember-gateway on-call runbook

Page the on-call engineer for ember-gateway when the burn rate exceeds the threshold
or when the service has been in a degraded state for fifteen minutes.

## Alerting

The primary alert fires when error rate stays above the objective for ten
minutes. A secondary alert fires when replica lag exceeds 120 seconds.

## First response

Confirm the alert against the dashboard before taking action. If queue depth is
above 2500, scale the gateway tier first. Do not restart the
deployment to clear a latency alert; the restart is more disruptive than the
symptom.
