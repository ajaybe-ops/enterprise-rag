# ADR 0003 — LLM-as-judge is not a gate

**Status:** accepted

Referenced from `app/evaluation/probe.py` and `app/evaluation/metrics.py`.

## Context

"Does this retrieved evidence actually answer the question?" is not something
string matching can determine. An LLM judge is the obvious tool, and it was the
obvious tool in the previous version of this project.

A judge score, however, is not reproducible: it changes with model version,
provider prompt handling, and sampling. Used as a regression gate it produces
false alarms, and a gate with a high false-alarm rate is disabled within a week.

## Decision

Judge-based scoring lives in a **separate, explicitly non-gating suite**. It may
run, report, and inform; it never sets a CI exit code and never blocks a merge.

The gating suite is structural (ADR 0001). The two suites are never combined into
a single score, because a weighted blend of a reproducible metric and a
non-reproducible one is a non-reproducible metric with extra steps.

## Consequences

- The gate stays trustworthy, which is the only property that makes it worth
  having.
- A genuinely wrong-but-retrievable answer will not be caught by the gate. The
  judge suite is the only thing that can see that class of failure, so it has to
  actually be run — which means someone has to choose to run it. A non-gating
  suite nobody runs is worthless, and this ADR does not solve that.
- If a judge is pinned to a fixed model version with temperature 0, it can be
  promoted to gating. That promotion is a new ADR, not a config change.
