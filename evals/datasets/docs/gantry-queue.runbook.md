---
version: 2.0
updated: 2026-08-17
owner: team-router
status: current
tenant: public
---
# gantry-queue on-call runbook

Page the on-call engineer for gantry-queue when the burn rate exceeds the threshold
or when the service has been in a degraded state for fifteen minutes.

## Alerting

The primary alert fires when error rate stays above the objective for ten
minutes. A secondary alert fires when replica lag exceeds 120 seconds.

## First response

Confirm the alert against the dashboard before taking action. If queue depth is
above 500, scale the router tier first. Do not restart the
deployment to clear a latency alert; the restart is more disruptive than the
symptom.
