"""Regression-direction tests.

`MetricDelta` is the gate that decides whether a change is shipped. A gate that
gets the sign wrong does not fail loudly -- it reports "healthy" while the thing
it was meant to catch gets worse. The original implementation applied
`delta < -tolerance` uniformly, so latency increases and leak-count increases
were invisible: the two failures a RAG regression gate exists to catch were the
two it could not see.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.regression import MetricDelta


class TestQualityMetricsAreHigherIsBetter:
    """A drop in a quality metric is a regression."""

    @pytest.mark.parametrize("name", ["pass_rate", "recall_at_5", "ndcg_at_5", "mrr", "fact_coverage"])
    def test_decrease_is_flagged(self, name):
        delta = MetricDelta(name, baseline=0.90, candidate=0.70)
        assert delta.regressed is True

    @pytest.mark.parametrize("name", ["pass_rate", "recall_at_5", "ndcg_at_5", "mrr", "fact_coverage"])
    def test_increase_is_not_flagged(self, name):
        delta = MetricDelta(name, baseline=0.70, candidate=0.90)
        assert delta.regressed is False

    def test_no_change_is_not_flagged(self):
        assert MetricDelta("pass_rate", 0.85, 0.85).regressed is False


class TestCostMetricsAreLowerIsBetter:
    """An increase in latency or leakage is a regression.

    This is the class of bug the uniform-sign implementation missed entirely.
    """

    def test_latency_increase_is_flagged(self):
        delta = MetricDelta(
            "p95_latency_ms", baseline=48.0, candidate=165.0, tolerance=1.0, higher_is_better=False
        )
        assert delta.delta == pytest.approx(117.0)
        assert delta.regressed is True

    def test_latency_decrease_is_not_flagged(self):
        delta = MetricDelta(
            "p95_latency_ms", baseline=165.0, candidate=48.0, tolerance=1.0, higher_is_better=False
        )
        assert delta.regressed is False

    def test_leakage_increase_is_flagged(self):
        delta = MetricDelta("leakage_count", baseline=0.0, candidate=3.0, higher_is_better=False)
        assert delta.regressed is True

    def test_leakage_decrease_is_not_flagged(self):
        delta = MetricDelta("leakage_count", baseline=3.0, candidate=0.0, higher_is_better=False)
        assert delta.regressed is False


class TestTolerance:
    def test_jitter_within_tolerance_is_ignored(self):
        # Sub-millisecond variance on a deterministic pipeline is measurement
        # noise. A gate that fires on it trains people to ignore the gate.
        delta = MetricDelta("p95_latency_ms", 48.0, 48.4, tolerance=1.0, higher_is_better=False)
        assert delta.regressed is False

    def test_change_beyond_tolerance_is_flagged(self):
        delta = MetricDelta("p95_latency_ms", 48.0, 50.5, tolerance=1.0, higher_is_better=False)
        assert delta.regressed is True

    def test_zero_tolerance_flags_any_worsening(self):
        assert MetricDelta("leakage_count", 0.0, 1.0, higher_is_better=False).regressed is True

    def test_tolerance_is_symmetric_in_magnitude(self):
        worse = MetricDelta("m", 1.0, 1.5, tolerance=0.25, higher_is_better=False)
        better = MetricDelta("m", 1.5, 1.0, tolerance=0.25, higher_is_better=False)
        assert worse.regressed is True
        assert better.regressed is False


class TestSerialisation:
    def test_direction_is_recorded_in_output(self):
        payload = MetricDelta("leakage_count", 0.0, 2.0, higher_is_better=False).to_dict()
        assert payload["regressed"] is True
        assert payload["higher_is_better"] is False

    def test_dict_round_trips_the_verdict(self):
        delta = MetricDelta("pass_rate", 0.9, 0.8)
        assert delta.to_dict()["regressed"] is delta.regressed
