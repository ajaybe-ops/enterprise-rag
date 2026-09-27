---
version: 2.4
updated: 2026-05-14
owner: team-compactor
status: current
tenant: public
---
# delta-compactor service level objectives

These are the contractual objectives for delta-compactor. They are reviewed quarterly
and deviations are reported to the owning team.

## Latency

The p99 request latency objective for delta-compactor is 600 milliseconds, measured
at the edge over a rolling 28-day window. A breach is defined as p99 exceeding
the objective for more than one hour.

## Availability

The availability objective is 99.9 percent measured monthly. Planned maintenance
is excluded from the calculation only when announced at least 72 hours ahead.

## Error budget

The error budget for delta-compactor is 0.1 percent of requests. When the
budget is exhausted, feature work is halted and reliability work is prioritised.
