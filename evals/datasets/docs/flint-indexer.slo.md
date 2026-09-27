---
version: 1.6
updated: 2026-07-16
owner: team-ingest
status: current
tenant: public
---
# flint-indexer service level objectives

These are the contractual objectives for flint-indexer. They are reviewed quarterly
and deviations are reported to the owning team.

## Latency

The p99 request latency objective for flint-indexer is 320 milliseconds, measured
at the edge over a rolling 28-day window. A breach is defined as p99 exceeding
the objective for more than one hour.

## Availability

The availability objective is 99.9 percent measured monthly. Planned maintenance
is excluded from the calculation only when announced at least 72 hours ahead.

## Error budget

The error budget for flint-indexer is 2.5 percent of requests. When the
budget is exhausted, feature work is halted and reliability work is prioritised.
