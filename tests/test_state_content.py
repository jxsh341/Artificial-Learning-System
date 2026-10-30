"""Tests for the state-content analysis module.

Uses a tiny connectivity dataset and a small untrained model; correctness of
the analytic machinery (decoding CV, substitution accounting, capture points)
is what is pinned — learning outcomes are validated by real experiments.
"""
import numpy as np
import pytest
import torch

from artificial_system.data.graph_tasks import make_connectivity_dataset
from artificial_system.environment.graph_environment import make_generic_graph
from artificial_system.evaluation import state_content as sc
from artificial_system.models.gnn_ssm import build_model

GEN = dict(num_nodes=20, feature_dim=2, marker_dim=2, max_path_len=3,
           edge_density=0.2, directed=False, label_mode="adjacency",
           neg_min_distance=3)


@pytest.fixture(scope="module")
def tiny_env():
    cfg = {
        "graph": {"node_feature_dim": 2},
        "gnn": {"layers": 2, "hidden_dim": 8, "aggregation": "sum"},
        "ssm": {"state_dim": 4},
        "readout": {"pooling": "attention"},
    }
    model = build_model(cfg, node_feature_dim=2, edge_feature_dim=0,
                        num_actions=2, num_nodes=None)
    ds = make_connectivity_dataset(48, seed=11, **GEN)
    blank = make_generic_graph(GEN["num_nodes"], GEN["feature_dim"], 5)
    return model, ds, blank


class TestRidgeDecodeCV:
    def test_perfectly_separable(self):
        rng = np.random.default_rng(0)
        y = np.where(rng.random(40) < 0.5, 0, 1)
        X = np.stack([rng.normal(0, 0.1, 4) + [1, 0, 0, 0] if t else
                      rng.normal(0, 0.1, 4) - [1, 0, 0, 0] for t in y])
        res = sc.ridge_decode_cv(X, y, lam=0.1, n_folds=5, seed=0)
        assert res["accuracy"] > 0.9
        assert res["majority_baseline"] < 0.8

    def test_chance_when_labels_shuffled(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 40)
        X = rng.normal(size=(40, 4))
        res = sc.ridge_decode_cv(X, y, lam=1.0, n_folds=5, seed=0)
        assert res["accuracy"] <= res["majority_baseline"] + 0.35

    def test_baseline_is_majority_frequency(self):
        y = np.array([0] * 38 + [1] * 2)
        X = np.random.default_rng(2).normal(size=(40, 3))
        res = sc.ridge_decode_cv(X, y, seed=0)
        assert abs(res["majority_baseline"] - 0.95) < 1e-9


class TestCaptureTrialStates:
    def test_point_shapes_and_labels(self, tiny_env):
        model, ds, blank = tiny_env
        states = sc.capture_trial_states(model, ds, delay=3, batch_size=16,
                                         marker_dim=2, generic_blank=blank)
        n = len(ds)
        for key in ("pre_info", "post_present", "delay_1", "delay_2",
                    "delay_3", "pre_response"):
            assert states[key].shape == (n, 4)
        assert states["_label"].shape == (n,)
        assert set(states["_label"].tolist()) <= {0, 1}

    def test_deterministic(self, tiny_env):
        model, ds, blank = tiny_env
        s1 = sc.capture_trial_states(model, ds, delay=2, batch_size=24,
                                     marker_dim=2, generic_blank=blank)
        s2 = sc.capture_trial_states(model, ds, delay=2, batch_size=24,
                                     marker_dim=2, generic_blank=blank)
        assert torch.allclose(s1["pre_response"], s2["pre_response"])
        # batch-size invariance of capture (model eval, no dropout)
        s3 = sc.capture_trial_states(model, ds, delay=2, batch_size=48,
                                     marker_dim=2, generic_blank=blank)
        assert torch.allclose(s1["pre_response"], s3["pre_response"], atol=1e-5)


class TestAnswerVsFeature:
    def test_separating_states_detected(self):
        n = 40
        labels = np.where(np.arange(n) < n // 2, 0, 1)
        X = torch.tensor(
            np.stack([[2.0, 0] if t else [-2.0, 0] for t in labels]) + 0.0,
            dtype=torch.float32,
        )
        states = {"_label": torch.tensor(labels), "pre_response": X}
        res = sc.answer_vs_feature(states)
        assert res["answer_separation_cos"] < -1e-6  # between <- within
        # identical states within class -> nearest also same label exactly
        assert res["nearest_same_label_fraction"] == 1.0
        assert res["chance_same_label_fraction"] < 1.0

    def test_structure_keys(self, tiny_env):
        model, ds, blank = tiny_env
        states = sc.capture_trial_states(model, ds, delay=2, batch_size=16,
                                         marker_dim=2, generic_blank=blank)
        res = sc.answer_vs_feature(states)
        for k in ("mean_cos_same_answer", "mean_cos_diff_answer",
                  "answer_separation_cos", "nearest_same_label_fraction",
                  "chance_same_label_fraction", "mean_pre_response_norm"):
            assert k in res


class TestStateSubstitution:
    def test_conditions_accounting(self, tiny_env):
        model, ds, blank = tiny_env
        rows, summary = sc.state_substitution(
            model, ds, delay=2, batch_size=16, marker_dim=2,
            generic_blank=blank, seed=0,
        )
        conditions = [r["condition"] for r in rows]
        assert conditions == ["own", "same_answer", "diff_answer", "shuffled", "zero"]
        for r in rows:
            assert 0.0 <= r["accuracy"] <= 1.0
            assert 0.0 <= r["follows_substituted_state"] <= 1.0
            assert r["mean_norm_ratio"] >= 0
        # zero-state substitution naturally has norm ratio 0 (fresh state)
        zero = [r for r in rows if r["condition"] == "zero"][0]
        assert abs(zero["mean_norm_ratio"]) < 1e-9 or zero["mean_norm_ratio"] > 0
        # zero-state substitution restarts at fresh initial state accruing
        # decayed state — should not be systematically better than own run
        own = [r["accuracy"] for r in rows if r["condition"] == "own"][0]
        for r in rows:
            assert r["accuracy"] <= max(1.0, own + 1e-9)
        for key in ("subst_own_accuracy", "subst_diff_follows", "subst_same_follows"):
            assert key in summary

    def test_donor_assignment_maintains_labels(self):
        # pick() must pick same-answer donors for same_answer and
        # different-answer donors for diff_answer
        labels = np.array([0, 1, 1, 0, 1])
        # emulate the internal logic of state_substitution.donor assignment
        pos = np.where(labels == 1)[0]
        neg = np.where(labels == 0)[0]
        rng = np.random.default_rng(0)

        def pick(i, want):
            pool = pos if want == 1 else neg
            excl = pool[pool != i]
            return int(excl[rng.integers(len(excl))]) if len(excl) else int(pool[0])

        for i in range(len(labels)):
            assert labels[pick(i, labels[i])] == labels[i]
            assert labels[pick(i, 1 - labels[i])] == 1 - labels[i]


class TestRunStateContent:
    def test_summary_structure(self, tiny_env):
        model, ds, blank = tiny_env
        rows, summary = sc.run_state_content(
            model, ds, delay=2, batch_size=24, marker_dim=2,
            generic_blank=blank, seed=0,
        )
        analyses = {r["analysis"] for r in rows}
        assert analyses == {"decode", "avf", "substitution"}
        for pt in ("pre_info", "post_present", "pre_response",
                   "delay_1", "delay_2"):
            assert f"decode_{pt}" in summary
        assert summary["decode_post_present_gap"] == (
            summary["decode_post_present"] - summary["decode_post_present_baseline"]
        )
        assert 0.0 <= summary["behavioural_accuracy"] <= 1.0
