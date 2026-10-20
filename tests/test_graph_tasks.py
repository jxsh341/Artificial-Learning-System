"""Graph-dependent task tests: dataset controls, environment, evaluations."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import torch

from artificial_system.data.graph_tasks import (
    ConnectivityDataset,
    make_connectivity_dataset,
    recompute_label,
)
from artificial_system.data.loaders import collate_graphs
from artificial_system.environment.graph_environment import (
    IGNORE_INDEX,
    GraphClassificationEnvironment,
)
from artificial_system.evaluation.graph_classification import (
    probe_dataset,
    run_graph_locus,
    run_graph_generalization,
    run_rewired_control,
)
from artificial_system.models import build_model


def config(**overrides) -> dict:
    base = {
        "gnn": {
            "layers": 3,
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
            "type": "graph_classification",
            "num_nodes": 30,
            "max_path_len": 3,
            "edge_density": 0.08,
            "marker_dim": 2,
            "directed": False,
            "delay": 1,
            "reward_correct": 1.0,
            "reward_incorrect": 0.0,
            "stochastic_prob": 0.0,
        },
        "training": {"batch_size": 4, "sequence_length": 8, "grad_clip": 1.0},
    }
    base["task"].update(overrides)
    return base


@pytest.fixture(scope="module")
def dataset():
    return make_connectivity_dataset(
        num_instances=40, num_nodes=30, feature_dim=8, marker_dim=2,
        max_path_len=3, edge_density=0.08, seed=0,
    )


def task_config(**overrides) -> dict:
    """Flat task-level config dict, as the environment expects."""
    return config(**overrides)["task"]


class TestDatasetControls:
    def test_all_instances_valid(self, dataset):
        for g in dataset.graphs:
            g.validate()
        assert dataset.labels.shape == (40,)

    def test_labels_balanced_and_binary(self, dataset):
        vals = set(dataset.labels.tolist())
        assert vals <= {0, 1}
        frac = float((dataset.labels == 1).float().mean())
        assert 0.25 <= frac <= 0.75

    def test_positives_connected_within_max_path_len(self, dataset):
        for g, lab in zip(dataset.graphs, dataset.labels):
            if int(lab) == 1:
                assert recompute_label(g, g.metadata["marked"], 3) == 1
            else:
                assert recompute_label(g, g.metadata["marked"], 3) == 0

    def test_negatives_distant_in_same_component(self, dataset):
        for g, lab in zip(dataset.graphs, dataset.labels):
            if int(lab) == 0:
                u, v = g.metadata["marked"]
                und = nx.Graph()
                und.add_nodes_from(range(g.num_nodes))
                und.add_edges_from({(int(a), int(b)) for a, b in g.edge_index.t().tolist()})
                assert nx.is_connected(und)  # same connected graph family
                assert nx.shortest_path_length(und, u, v) > 3

    def test_adjacency_mode_label_depends_on_marked_pair_only(self):
        d = make_connectivity_dataset(
            30, num_nodes=40, feature_dim=8, marker_dim=2,
            edge_density=0.06, seed=2, label_mode="adjacency", neg_min_distance=3,
        )
        for g, lab in zip(d.graphs, d.labels):
            und = nx.Graph()
            und.add_nodes_from(range(g.num_nodes))
            und.add_edges_from({(int(a), int(b)) for a, b in g.edge_index.t().tolist()})
            assert nx.is_connected(und)  # identical graph-level structure across classes
            u, v = g.metadata["marked"]
            d_ij = nx.shortest_path_length(und, u, v)
            if int(lab) == 1:
                assert d_ij == 1
            else:
                assert d_ij >= 3
            assert recompute_label(g, g.metadata["marked"], 3, "adjacency") == int(lab)

    def test_edge_counts_matched_across_classes(self, dataset):
        stats = dataset.metadata["stats"]
        pos, neg = stats["mean_edges_positive"], stats["mean_edges_negative"]
        assert abs(pos - neg) / max(pos, 1) < 0.25

    def test_marker_channel_clean_and_placed(self, dataset):
        for g in dataset.graphs:
            marked = g.metadata["marked"]
            for node in range(g.num_nodes):
                marker = g.node_features[node, :2]
                if node in marked:
                    assert marker[marked.index(node)] == 1.0
                else:
                    assert torch.equal(marker, torch.zeros(2))

    def test_reproducible_with_seed(self):
        d1 = make_connectivity_dataset(10, num_nodes=30, feature_dim=8, seed=5)
        d2 = make_connectivity_dataset(10, num_nodes=30, feature_dim=8, seed=5)
        assert d1.labels.tolist() == d2.labels.tolist()
        for g1, g2 in zip(d1.graphs, d2.graphs):
            assert torch.equal(g1.edge_index, g2.edge_index)

    def test_recompute_label_after_disconnection(self, dataset):
        from artificial_system.data.perturbations import remove_edges

        g = dataset.graphs[[i for i, l in enumerate(dataset.labels) if int(l) == 1][0]]
        # remove ALL edges -> definitely disconnected -> label 0
        stripped = remove_edges(g, 1.0, seed=0)
        stripped = type(g)(
            node_features=stripped.node_features,
            edge_index=stripped.edge_index[:, :0],
        )
        assert recompute_label(stripped, g.metadata["marked"], 3) == 0


class TestEnvironment:
    def test_reset_batched_variable_sizes(self, dataset):
        env = GraphClassificationEnvironment(dataset, task_config(), seed=0)
        obs = env.reset(4)
        assert obs.num_graphs == 4
        assert obs.batch.max() == 3

    def test_delay_masks_targets_then_labels(self, dataset):
        env = GraphClassificationEnvironment(dataset, task_config(delay=2), seed=0)
        env.reset(4)
        assert torch.all(env.current_targets() == IGNORE_INDEX)
        idxs = env._idxs.clone()
        env.step(torch.zeros(4, dtype=torch.long))
        env.step(torch.zeros(4, dtype=torch.long))
        assert torch.equal(env.current_targets(), env.labels_for(idxs))

    def test_blank_observation_clears_markers(self, dataset):
        env = GraphClassificationEnvironment(dataset, task_config(), seed=0)
        env.reset(2)
        blank = env.blank_observation()
        assert torch.equal(blank.node_features[:, :2], torch.zeros(blank.num_nodes, 2))
        kept = env.blank_observation(keep_markers=True)
        assert not torch.equal(kept.node_features[:, :2], torch.zeros(kept.num_nodes, 2))

    def test_step_observation_blanked_during_delay(self, dataset):
        """Regression: delay and response steps must NOT show the markers."""
        env = GraphClassificationEnvironment(dataset, task_config(delay=2), seed=0)
        obs0 = env.reset(2)
        assert not torch.equal(obs0.node_features[:, :2], torch.zeros(obs0.num_nodes, 2))
        obs1, _, _, _ = env.step(torch.zeros(2, dtype=torch.long))
        assert torch.equal(obs1.node_features[:, :2], torch.zeros(obs1.num_nodes, 2))
        obs2, _, _, _ = env.step(torch.zeros(2, dtype=torch.long))
        assert torch.equal(obs2.node_features[:, :2], torch.zeros(obs2.num_nodes, 2))

    def test_reward_follows_labels(self, dataset):
        env = GraphClassificationEnvironment(dataset, task_config(delay=0), seed=0)
        env.reset(4)
        targets = env.current_targets()
        _, rewards, _, _ = env.step(targets.clone())
        assert torch.all(rewards == 1.0)

    def test_determinism(self, dataset):
        e1 = GraphClassificationEnvironment(dataset, task_config(), seed=3)
        e2 = GraphClassificationEnvironment(dataset, task_config(), seed=3)
        e1.reset(4)
        e2.reset(4)
        assert torch.equal(e1._idxs, e2._idxs)


class TestEvaluations:
    def test_probe_dataset_shapes(self, dataset):
        model = build_model(config(), node_feature_dim=8, num_actions=2, num_nodes=None)
        out = probe_dataset(model, dataset, delay=0, batch_size=8)
        assert 0.0 <= out["accuracy"] <= 1.0
        assert set(out["accuracy_by_class"]) == {"0", "1"}

    def test_probe_delayed(self, dataset):
        model = build_model(config(), node_feature_dim=8, num_actions=2, num_nodes=None)
        out = probe_dataset(model, dataset, delay=2, batch_size=8)
        assert 0.0 <= out["accuracy"] <= 1.0

    def test_rewired_control_rows(self, dataset):
        model = build_model(config(), node_feature_dim=8, num_actions=2, num_nodes=None)
        rows, summary = run_rewired_control(model, dataset, levels=[0.5, 1.0], batch_size=8)
        conditions = {(r["condition"], r["level"]) for r in rows}
        assert ("intact", 0.0) in conditions
        assert ("rewired_relabeled", 1.0) in conditions
        assert ("rewired_original_labels", 1.0) in conditions
        assert ("swapped_relabeled", 1.0) in conditions
        assert ("swapped_original_labels", 1.0) in conditions

    def test_graph_locus_conditions(self, dataset):
        model = build_model(config(), node_feature_dim=8, num_actions=2, num_nodes=None)
        rows, summary = run_graph_locus(model, dataset, delay=2, batch_size=8)
        keys = {r["condition"] for r in rows}
        assert keys == {
            "immediate", "delayed_retain", "delayed_reset_cleared",
            "delayed_markers_kept", "delayed_markers_kept_reset",
        }
        assert all(0.0 <= v <= 1.0 for v in summary.values())

    def test_unseen_size_generalization(self, dataset):
        unseen = make_connectivity_dataset(
            20, num_nodes=60, feature_dim=8, marker_dim=2, max_path_len=3,
            edge_density=0.08, seed=9,
        )
        model = build_model(config(), node_feature_dim=8, num_actions=2, num_nodes=None)
        rows, summary = run_graph_generalization(model, dataset, unseen_size_dataset=unseen, batch_size=8)
        splits = {r["split"] for r in rows}
        assert splits == {"test_unseen_instances", "test_unseen_size"}

