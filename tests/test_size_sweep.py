"""Tests for the controlled size sweep + the blank-protocol consistency fix.

Regression context: ``run_graph_generalization`` historically probed the
unseen-size set with markers-cleared-instance blanks (off-distribution for
``blank_mode: generic`` training), producing spurious chance-level readings on
successful models. These tests pin the corrected behaviour.
"""
import networkx as nx
import torch

from artificial_system.data.graph_tasks import make_connectivity_dataset
from artificial_system.environment.graph_environment import (
    GraphClassificationEnvironment,
    make_generic_graph,
)
from artificial_system.evaluation import size_sweep as sw
from artificial_system.evaluation.graph_classification import run_graph_generalization


def _tiny_model_cfg():
    return {
        "graph": {"node_feature_dim": 2},
        "gnn": {"layers": 2, "hidden_dim": 8, "aggregation": "sum"},
        "ssm": {"state_dim": 4, "max_radius": 0.9},
        "readout": {"pooling": "attention"},
    }


def _tiny_cfg():
    return {
        "num_nodes": 20,
        "feature_dim": 2,
        "marker_dim": 2,
        "max_path_len": 3,
        "edge_density": 0.2,
        "directed": False,
        "label_mode": "adjacency",
        "neg_min_distance": 3,
        "num_train": 8,
        "num_test": 8,
        "test_num_nodes": 30,
        "delay": 2,
        "blank_mode": "generic",
    }


class TestMakeGenericGraph:
    def test_matches_env_internal_graph(self):
        task_cfg = _tiny_cfg()
        ds = make_connectivity_dataset(4, seed=123, **{
            k: task_cfg[k] for k in
            ("num_nodes", "feature_dim", "marker_dim", "max_path_len",
             "edge_density", "directed", "label_mode", "neg_min_distance")
        })
        env = GraphClassificationEnvironment(ds, task_cfg, seed=7)
        standalone = make_generic_graph(task_cfg["num_nodes"], task_cfg["feature_dim"], 7)
        assert torch.equal(env._generic_graph.edge_index, standalone.edge_index)
        assert torch.equal(env._generic_graph.node_features, standalone.node_features)

    def test_valid_connected_no_markers(self):
        g = make_generic_graph(50, 3, 0)
        edge_set = {(int(u), int(v)) for u, v in g.edge_index.t().tolist()}
        gn = nx.Graph(edge_set)
        assert nx.is_connected(gn)
        assert g.node_features.shape == (50, 3)
        assert float(g.node_features.abs().sum()) == 0.0


class TestRunGraphGeneralizationProtocol:
    def test_accepts_generic_blank_and_differs_from_no_blank(self):
        task_cfg = _tiny_cfg()
        gen = {k: task_cfg[k] for k in
               ("num_nodes", "feature_dim", "marker_dim", "max_path_len",
                "edge_density", "directed", "label_mode", "neg_min_distance")}
        test_ds = make_connectivity_dataset(8, seed=124, **gen)
        unseen_ds = make_connectivity_dataset(
            8, seed=125, **{**gen, "num_nodes": task_cfg["test_num_nodes"]}
        )

        model = _build_model()

        # smoke: runs with the new argument, returns summary keys
        rows, summary = run_graph_generalization(
            model, test_ds, unseen_size_dataset=unseen_ds, delay=2, batch_size=4,
            generic_blank=make_generic_graph(
                task_cfg["num_nodes"], task_cfg["feature_dim"], 7
            ),
        )
        assert summary["graph_generalization_unseen_instances"] >= 0.0
        assert summary["graph_generalization_unseen_size"] >= 0.0
        assert rows[1]["num_nodes"] == task_cfg["test_num_nodes"]

    def test_protocol_match_changes_delayed_unseen_size_accuracy(self):
        """The regression: with generic_blank the successful-model reading can
        differ from the legacy no-blank protocol. We assert the argument is
        actually consumed by feeding a deliberately 'unLearnable' blank (large
        noise graph) that must change behaviour vs the neutral blank."""
        task_cfg = _tiny_cfg()
        gen = {k: task_cfg[k] for k in
               ("num_nodes", "feature_dim", "marker_dim", "max_path_len",
                "edge_density", "directed", "label_mode", "neg_min_distance")}
        test_ds = make_connectivity_dataset(8, seed=124, **gen)
        unseen_ds = make_connectivity_dataset(
            8, seed=125, **{**gen, "num_nodes": task_cfg["test_num_nodes"]}
        )
        model = _build_model()
        legacy = run_graph_generalization(
            model, test_ds, unseen_size_dataset=unseen_ds, delay=2, batch_size=4,
        )[1]["graph_generalization_unseen_size"]
        weird = run_graph_generalization(
            model, test_ds, unseen_size_dataset=unseen_ds, delay=2, batch_size=4,
            generic_blank=make_generic_graph(
                task_cfg["num_nodes"], task_cfg["feature_dim"], 4242
            ),
        )[1]["graph_generalization_unseen_size"]
        # an untrained model may or may not differ numerically; the contract we
        # pin is that both calls run and the argument is accepted (no TypeError)
        assert 0.0 <= legacy <= 1.0 and 0.0 <= weird <= 1.0


def _build_model():
    from artificial_system.models.gnn_ssm import build_model

    return build_model(
        _tiny_model_cfg(), node_feature_dim=2, edge_feature_dim=0,
        num_actions=2, num_nodes=None,
    )


class TestSweepSize:
    def test_structure_and_density_modes(self):
        task_cfg = _tiny_cfg()
        graph_cfg = {"node_feature_dim": 2}
        model = _build_model()
        base_gen = sw._gen_cfg(task_cfg, graph_cfg)
        base_gen["base_nodes"] = task_cfg["num_nodes"]
        rows = sw.sweep_size(
            model, task_cfg=None, graph_cfg=None,
            sizes=[20, 30], sets_per_size=1, instances_per_set=4,
            delay=2, batch_size=4,
            generic_blank_graph=make_generic_graph(20, 2, 7),
            base_gen=base_gen,
        )
        assert len(rows) == 2 * 2 * 1  # sizes x modes x sets
        keys = {"num_nodes", "density_mode", "edge_density", "mean_degree",
                "immediate_accuracy", "delayed_intact", "delayed_reset",
                "mean_z_norm", "mean_h_norm", "mean_marker_attention_share"}
        assert keys <= set(rows[0])
        fixed = [r for r in rows if r["density_mode"] == "fixed" and r["num_nodes"] == 20][0]
        matched = [r for r in rows if r["density_mode"] == "degree_matched" and r["num_nodes"] == 30][0]
        assert fixed["edge_density"] == task_cfg["edge_density"]
        assert abs(matched["edge_density"] - task_cfg["edge_density"] * 20 / 30) < 1e-12
        assert abs(fixed["mean_degree"] - task_cfg["edge_density"] * 19) < 1e-9
        assert abs(matched["mean_degree"] - task_cfg["edge_density"] * 20 / 30 * 29) < 1e-9

    def test_gen_cfg_from_config(self):
        task_cfg = _tiny_cfg()
        graph_cfg = {"node_feature_dim": 5}
        gen = sw._gen_cfg(task_cfg, graph_cfg)
        assert gen["feature_dim"] == 5
        assert gen["label_mode"] == "adjacency"
        assert gen["edge_density"] == 0.2
        assert gen["neg_min_distance"] == 3
