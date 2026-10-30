"""Tests for the H5 coupling-rescue machinery (report §8b).

Freeze semantics, label-shuffling control, and dataset reconstruction are
pinned; the full intervention is validated by the real run, not here.
"""
import numpy as np
import torch

from artificial_system.data.graph_tasks import make_connectivity_dataset
from artificial_system.experiments import coupling_rescue as cr
from artificial_system.models.gnn_ssm import build_model

GEN = dict(num_nodes=20, feature_dim=2, marker_dim=2, max_path_len=3,
           edge_density=0.2, directed=False, label_mode="adjacency",
           neg_min_distance=3)


def _model():
    cfg = {
        "graph": {"node_feature_dim": 2},
        "gnn": {"layers": 2, "hidden_dim": 8, "aggregation": "sum"},
        "ssm": {"state_dim": 4},
        "readout": {"pooling": "attention"},
    }
    return build_model(cfg, node_feature_dim=2, edge_feature_dim=0,
                       num_actions=2, num_nodes=None)


class TestSetTrainables:
    def test_readout_only_freezes_generation_path(self):
        model = _model()
        cr.set_trainables(model, "readout_only")
        for p in model.graph_encoder.parameters():
            assert not p.requires_grad
        for p in model.pooling.parameters():
            assert not p.requires_grad
        ssm = model.temporal_model
        assert not ssm.A_raw.requires_grad          # state decay (generation)
        assert not ssm.B.weight.requires_grad       # input->state (generation)
        assert ssm.C.weight.requires_grad           # state->output (readout)
        assert ssm.C.bias.requires_grad
        assert ssm.D.weight.requires_grad           # z shortcut (readout path)
        for p in model.head.parameters():
            assert p.requires_grad

    def test_all_restores(self):
        model = _model()
        cr.set_trainables(model, "readout_only")
        cr.set_trainables(model, "all")
        assert all(p.requires_grad for p in model.parameters())


class TestRelabelDataset:
    def test_shuffle_breaks_association_keeps_inputs(self):
        ds = make_connectivity_dataset(20, seed=3, **GEN)
        shuffled = cr.relabel_dataset(ds, seed=777)
        # same graphs, same multiset of labels, different assignment
        assert len(shuffled) == len(ds)
        assert set(shuffled.labels.tolist()) == set(ds.labels.tolist())
        assert not torch.equal(shuffled.labels, ds.labels)
        for g1, g2 in zip(ds.graphs, shuffled.graphs):
            assert torch.equal(g1.edge_index, g2.edge_index)
            assert torch.equal(g1.node_features, g2.node_features)

    def test_deterministic(self):
        ds = make_connectivity_dataset(12, seed=5, **GEN)
        a = cr.relabel_dataset(ds, seed=9)
        b = cr.relabel_dataset(ds, seed=9)
        assert torch.equal(a.labels, b.labels)


class TestBuildDatasets:
    def test_shapes_and_metadata(self):
        cfg = {
            "graph": {"node_feature_dim": 2},
            "task": dict(GEN, num_train=10, num_test=6, dataset_seed=123,
                         delay=2, blank_mode="generic", num_nodes=20,
                         edge_density=0.2),
        }
        d = cr.build_datasets_from_cfg(cfg)
        assert len(d["train"]) == 10 and len(d["test"]) == 6
        assert d["train"].graphs[0].num_nodes == 20
        assert d["train"].graphs[0].node_features.shape[1] == 2


class TestConfigureProbe:
    def test_quick_probe_keys(self):
        model = _model()
        ds = make_connectivity_dataset(8, seed=11, **GEN)
        from artificial_system.environment.graph_environment import make_generic_graph
        blank = make_generic_graph(GEN["num_nodes"], GEN["feature_dim"], 5)
        res = cr.quick_probe(model, ds, delay=3, marker_dim=2, generic_blank=blank)
        for k in ("immediate", "delayed_intact", "delayed_reset"):
            assert 0.0 <= res[k] <= 1.0

    def test_arm_summary_writes(self, tmp_path):
        """End-to-end smoke of one short arm on a tiny budget."""
        cfg = {
            "graph": {"node_feature_dim": 2},
            "gnn": {"layers": 2, "hidden_dim": 8, "aggregation": "sum"},
            "ssm": {"state_dim": 4},
            "readout": {"pooling": "attention"},
            "task": dict(GEN, num_train=12, num_test=8, dataset_seed=123,
                         delay=2, blank_mode="generic",
                         reward_correct=1.0, reward_incorrect=0.0,
                         stochastic_prob=0.0),
            "training": {"batch_size": 4, "sequence_length": 4,
                         "learning_rate": 1e-3, "weight_decay": 0.0,
                         "grad_clip": 1.0, "record_aux_every": 0},
        }
        datasets = cr.build_datasets_from_cfg(cfg)
        ck = {"model_state": _model().state_dict(), "seed": 0,
              "recorded_delayed": 0.5, "source_path": "test"}
        summary = cr.run_coupling_arm(
            cfg, datasets, ck, "treatment_a", tmp_path, seed=0,
            schedule_spec={"stage3": {"episodes": 2}, "stage4": {"episodes": 2}},
        )
        assert summary["arm"] == "treatment_a"
        assert "rubric" in summary and "pre_analysis" in summary
        assert (tmp_path / "training_probes.csv").exists()
        assert (tmp_path / "arm_summary.json").exists()
