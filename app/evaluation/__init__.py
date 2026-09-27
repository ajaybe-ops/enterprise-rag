"""Retrieval quality: contracts, deterministic metrics, and regression attribution.

The public surface is deliberately small. The point of the package is a claim a
reviewer can check -- "this corpus change caused these 3 regressions, and here
is the evidence" -- so the import list is the three things that make that claim:
declare a contract, measure it, compare two measurements.

    from app.evaluation import load_probe_set, run_probe_set, attribute
"""

from app.evaluation.metrics import (
    AggregateReport,
    aggregate,
    context_precision,
    evaluate_probe,
    fact_coverage,
    first_required_rank,
    leakage,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    source_recall,
)
from app.evaluation.probe import (
    PROBE_KINDS,
    Probe,
    ProbeKind,
    ProbeOutcome,
    ProbeSet,
    dataset_id_for,
    load_probe_set,
)
from app.evaluation.regression import (
    Attribution,
    Cause,
    MetricDelta,
    ProbeRegression,
    RegressionReport,
    attribute,
    compare,
)
from app.evaluation.runner import (
    ProbeExecution,
    ProbeExecutor,
    RetrievalExecutor,
    RunReport,
    config_fingerprint,
    run_probe_set,
)

__all__ = [
    "PROBE_KINDS",
    "AggregateReport",
    "Attribution",
    "Cause",
    "MetricDelta",
    "Probe",
    "ProbeExecution",
    "ProbeExecutor",
    "ProbeKind",
    "ProbeOutcome",
    "ProbeRegression",
    "ProbeSet",
    "RegressionReport",
    "RetrievalExecutor",
    "RunReport",
    "aggregate",
    "attribute",
    "compare",
    "config_fingerprint",
    "context_precision",
    "dataset_id_for",
    "evaluate_probe",
    "fact_coverage",
    "first_required_rank",
    "leakage",
    "load_probe_set",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "run_probe_set",
    "source_recall",
]
