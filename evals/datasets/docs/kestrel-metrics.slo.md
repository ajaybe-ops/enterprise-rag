---
version: 3.4
updated: 2026-03-12
owner: team-gateway
status: current
tenant: public
---
# kestrel-metrics service level objectives

These are the contractual objectives for kestrel-metrics. They are reviewed quarterly
and deviations are reported to the owning team.

## Latency

The p99 request latency objective for kestrel-metrics is 600 milliseconds, measured
at the edge over a rolling 28-day window. A breach is defined as p99 exceeding
the objective for more than one hour.

## Availability

The availability objective is 99.9 percent measured monthly. Planned maintenance
is excluded from the calculation only when announced at least 72 hours ahead.

## Error budget

The error budget for kestrel-metrics is 2.5 percent of requests. When the
budget is exhausted, feature work is halted and reliability work is prioritised.
