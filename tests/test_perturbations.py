"""Tests for graph perturbations, memory curve, and temporal attacks."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from artificial_system.data.perturbations import (
    GRAPH_PERTURBATIONS,
    drop_community,
    perturb_graph,
    perturb_weights,
    randomize_edges,
    remove_edges,
    remove_nodes,
    shuffle_features,
)
from artificial_system.data.schema import GraphData
from artificial_system.environment.cue_environment import CueLearningEnvironment
from artificial_system.evaluation.memory_curve import run_memory_curve
from artificial_system.evaluation.perturbations import (
    probe_task,
    run_temporal_perturbations,
    storage_locus,
)
from artificial_system.models import build_model


def tiny_config() -> dict:
    cfg = {
        "gnn": {
            "layers": 1,
            "hidden_dim": 16,
            "dropout": 0.0,
            "aggregation": "mean",
            "residual": True,
            "use_edge_features": False,
        },
        "ssm": {
            "state_dim": 16,
            "diagonal": True,
            "max_radius": 0.99,
            "init_decay": 0.9,
            "nonlinearity": "tanh",
        },
        "readout": {"pooling": "mean", "head_hidden": 0},
        "task": {
            "num_cues": 4,
            "num_actions": 2,
            "cue_feature_dim": 4,
            "structure": "onehot",
            "mapping": "cyclic",
            "delay": 1,
            "reward_correct": 1.0,
            "reward_incorrect": 0.0,
            "stochastic_prob": 0.0,
        },
        "training": {"batch_size": 4, "sequence_length": 8, "grad_clip": 1.0},
    }
    return cfg


@pytest.fixture()
def model_cfg():
    cfg = tiny_config()
    cfg["gnn"].update({"hidden_dim": 16, "layers": 1})
    cfg["ssm"].update({"state_dim": 16})
    return cfg


@pytest.fixture()
def trained_model(model_cfg, small_graph):
    torch.manual_seed(0)
    return build_model(model_cfg, node_feature_dim=4, num_actions=2, num_nodes=small_graph.num_nodes)


@pytest.fixture()
def env(small_graph):
    return CueLearningEnvironment(small_graph, tiny_config()["task"], seed=0)


class TestGraphPerturbations:
    @pytest.mark.parametrize("kind", GRAPH_PERTURBATIONS)
    def test_all_perturbations_produce_valid_graphs(self, small_graph, kind):
        g = perturb_graph(small_graph, kind, 0.5, seed=0)
        assert isinstance(g, GraphData)
        g.validate()  # raises on any structural violation
        assert g.node_features.shape == small_graph.node_features.shape

    def test_remove_edges_reduces_edge_count(self, small_graph):
        g = remove_edges(small_graph, 0.5, seed=0)
        assert 0 < g.num_edges <= small_graph.num_edges

    def test_randomize_edges_preserves_count(self, small_graph):
        g = randomize_edges(small_graph, 0.5, seed=0)
        assert g.num_edges == small_graph.num_edges

    def test_perturb_weights_deterministic_and_changes_values(self, small_graph):
        small_graph = GraphData(
            small_graph.node_features, small_graph.edge_index,
            edge_features=torch.rand(small_graph.num_edges, 2) + 0.5,
        )
        g1 = perturb_weights(small_graph, 0.5, seed=0)
        g2 = perturb_weights(small_graph, 0.5, seed=0)
        g3 = perturb_weights(small_graph, 0.5, seed=1)
        assert torch.allclose(g1.edge_features, g2.edge_features)
        assert not torch.allclose(g1.edge_features, g3.edge_features)

    def test_shuffle_features_preserves_multiset(self, small_graph):
        g = shuffle_features(small_graph, seed=3)
        a = small_graph.node_features.sort(dim=0).values
        b = g.node_features.sort(dim=0).values
        assert torch.allclose(a, b)

    def test_remove_nodes_sets_mask(self, small_graph):
        g = remove_nodes(small_graph, 0.25, seed=0)
        assert g.node_mask is not None
        n_silenced = int((~g.node_mask).sum())
        assert n_silenced == 2

    def test_drop_community_masks_some_nodes(self, small_graph):
        g = drop_community(small_graph, seed=0)
        assert 0 < int((~g.node_mask).sum()) < small_graph.num_nodes


class TestMemoryCurve:
    def test_rows_match_delays_and_finite(self, trained_model, env):
        env.reset(4)
        rows, summary = run_memory_curve(trained_model, env, delays=[0, 5, 20], repeats=4)
        assert [r["delay"] for r in rows] == [0, 5, 20]
        assert all(np.isfinite(r["accuracy"]) for r in rows)
        assert summary["memory_ceiling"] == rows[0]["accuracy"]
        assert "memory_half_life" in summary

    def test_repeats_shrink_for_large_delays(self, trained_model, env):
        env.reset(4)
        rows, _ = run_memory_curve(trained_model, env, delays=[10, 500], repeats=20, repeats_floor=2)
        assert rows[0]["repeats"] == 20
        assert rows[1]["repeats"] == 2


class TestTemporalPerturbations:
    def test_probe_task_returns_accuracy(self, trained_model, env):
        env.reset(4)
        rec = probe_task(trained_model, env, 20, batch_size=4)
        assert 0.0 <= rec["accuracy"] <= 1.0

    def test_run_temporal_perturbations_rows(self, trained_model, env):
        env.reset(4)
        rows, summary = run_temporal_perturbations(
            trained_model, env, num_steps=20, batch_size=4,
            noise_sigmas=[0.5], reset_fractions=[0.5], truncation_intervals=[3],
        )
        names = {(r["perturbation"], r["level"]) for r in rows}
        assert ("none", 0.0) in names
        assert ("state_reset", 1.0) in names
        assert ("partial_reset", 0.5) in names
        assert ("noise", 0.5) in names
        assert ("truncate", 3.0) in names
        assert summary["temporal_intact_accuracy"] == rows[0]["accuracy"]

    def test_storage_locus_conditions(self, trained_model, small_graph):
        def build_env(graph, seed, task_overrides=None):
            return CueLearningEnvironment(graph, {**tiny_config()["task"], **(task_overrides or {})}, seed=seed)

        rows, summary = storage_locus(trained_model, build_env, small_graph, num_steps=20, batch_size=4)
        keys = {r["locus"] for r in rows}
        assert keys == {
            "locus_full", "locus_no_state", "locus_no_graph", "locus_weights_only",
            "locus_immediate_full", "locus_immediate_no_state",
            "locus_immediate_no_graph", "locus_immediate_weights_only",
        }
        assert all(0.0 <= summary[k] <= 1.0 for k in keys if not k.endswith("_state_norm"))
