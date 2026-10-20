"""Metric and statistics tests (spec §22, §26)."""
from __future__ import annotations

import numpy as np

from artificial_system.evaluation.metrics import (
    curve_endpoints,
    internal_state_stats,
    residual_preference,
    retention_ratio,
    trials_to_criterion,
)
from artificial_system.evaluation.statistics import aggregate_summaries, confidence_interval


class TestMetrics:
    def test_curve_endpoints(self):
        v = np.linspace(0.0, 1.0, 100)
        first, last = curve_endpoints(v, frac=0.1)
        assert abs(first - 0.045) < 0.02
        assert abs(last - 0.955) < 0.02

    def test_curve_endpoints_nan_safe(self):
        v = np.array([np.nan, 0.5, np.nan, 1.0])
        first, last = curve_endpoints(v, frac=0.5)
        assert first == 0.5 and last == 1.0

    def test_trials_to_criterion_reached(self):
        acc = np.array([0.5] * 10 + [0.95] * 30)
        t = trials_to_criterion(acc, threshold=0.9, window=10)
        assert 10 < t <= 30

    def test_trials_to_criterion_never_reached(self):
        assert trials_to_criterion(np.full(50, 0.5), 0.9, 10) == -1

    def test_trials_to_criterion_immediate(self):
        assert trials_to_criterion(np.full(50, 1.0), 0.9, 10) == 1

    def test_retention_ratio(self):
        assert retention_ratio(0.8, 1.0) == 0.8
        assert np.isnan(retention_ratio(0.5, 0.0))

    def test_residual_preference(self):
        actions = np.array([0, 0, 1, 1])
        cues = np.array([0, 1, 2, 3])
        old = np.array([0, 0, 1, 0])  # matches 3 of 4
        assert residual_preference(actions, cues, old) == 0.75

    def test_internal_state_stats(self):
        rng = np.random.default_rng(0)
        states = rng.standard_normal((20, 4, 8))
        stats = internal_state_stats(states)
        assert stats["state_norm_mean"] > 0
        assert -1.0 <= stats["mean_pairwise_cosine"] <= 1.0


class TestStatistics:
    def test_confidence_interval(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean, half = confidence_interval(values)
        assert mean == 3.0
        assert half > 0
        assert mean - half < 2.0 < mean < 4.0 < mean + half  # CI brackets the bulk of the data

    def test_confidence_interval_single_value(self):
        mean, half = confidence_interval([0.7])
        assert mean == 0.7 and np.isnan(half)

    def test_aggregate_summaries(self):
        summaries = [
            {"seed": 0, "acc": 0.8, "label": "a"},
            {"seed": 1, "acc": 0.9, "label": "b"},
            {"seed": 2, "acc": 1.0, "label": "c"},
        ]
        agg = aggregate_summaries(summaries)
        assert agg["n_seeds"] == 3
        assert abs(agg["acc_mean"] - 0.9) < 1e-9
        assert agg["acc_median"] == 0.9
        assert agg["acc_per_seed"] == [0.8, 0.9, 1.0]
        assert "label_mean" not in agg  # non-numeric keys skipped
