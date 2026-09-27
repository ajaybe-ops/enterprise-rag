---
version: 1.6
updated: 2026-07-16
owner: team-ingest
status: current
tenant: public
---
# flint-indexer specification

The flint-indexer service owns the ingest path for the platform. It is
deployed as a stateless deployment with horizontal autoscaling driven by queue
depth. Traffic enters through the internal mesh and is authenticated with a
service identity token.

## Capacity

The service is sized for a sustained queue depth of 500 messages per
replica. Autoscaling begins above that figure and the target maximum is four
replicas. Sustained depth above the target triggers a capacity review rather than
automatic scale-out, because the downstream ingest tier is the binding
constraint.

## Replication

Replica lag is monitored as a first-class signal. The alerting threshold is
60 seconds, and the service is considered degraded once lag exceeds
twice that value. Automatic failover is not enabled; recovery is operator-initiated
so that a bad configuration change is not replicated across the fleet.

## Configuration

All tunables live in the service config map. Restarting the deployment applies
changes. There is no hot-reload path for the ingest subsystem, and adding
one is tracked but not scheduled.
