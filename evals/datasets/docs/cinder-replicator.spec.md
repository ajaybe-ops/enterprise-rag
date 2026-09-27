---
version: 1.3
updated: 2026-04-13
owner: team-replicator
status: current
tenant: public
---
# cinder-replicator specification

The cinder-replicator service owns the replicator path for the platform. It is
deployed as a stateless deployment with horizontal autoscaling driven by queue
depth. Traffic enters through the internal mesh and is authenticated with a
service identity token.

## Capacity

The service is sized for a sustained queue depth of 1000 messages per
replica. Autoscaling begins above that figure and the target maximum is four
replicas. Sustained depth above the target triggers a capacity review rather than
automatic scale-out, because the downstream replicator tier is the binding
constraint.

## Replication

Replica lag is monitored as a first-class signal. The alerting threshold is
30 seconds, and the service is considered degraded once lag exceeds
twice that value. Automatic failover is not enabled; recovery is operator-initiated
so that a bad configuration change is not replicated across the fleet.

## Configuration

All tunables live in the service config map. Restarting the deployment applies
changes. There is no hot-reload path for the replicator subsystem, and adding
one is tracked but not scheduled.
