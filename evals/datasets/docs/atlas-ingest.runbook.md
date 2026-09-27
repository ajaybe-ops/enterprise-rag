---
version: 1.0
updated: 2026-01-10
owner: team-ingest
status: current
tenant: public
---
# atlas-ingest on-call runbook

Page the on-call engineer for atlas-ingest when the burn rate exceeds the threshold
or when the service has been in a degraded state for fifteen minutes.

## Alerting

The primary alert fires when error rate stays above the objective for ten
minutes. A secondary alert fires when replica lag exceeds 5 seconds.

## First response

Confirm the alert against the dashboard before taking action. If queue depth is
above 2500, scale the ingest tier first. Do not restart the
deployment to clear a latency alert; the restart is more disruptive than the
symptom.
