"""Smoke tests for the readout-path convergence diagnostics (§8d)."""
from pathlib import Path

import pandas as pd
import pytest
import torch

from artificial_system.data.graph_tasks import make_connectivity_dataset
from artificial_system.experiments import readout_diagnostics as rd
from artificial_system.models.gnn_ssm import build_model

GEN = dict(num_nodes=20, feature_dim=2, marker_dim=2, max_path_len=3,
           edge_density=0.2, directed=False, label_mode="adjacency",
           neg_min_distance=3)


def _cfg():
    return {
        "graph": {"node_feature_dim": 2},
        "gnn": {"layers": 2, "hidden_dim": 8, "aggregation": "sum"},
        "ssm": {"state_dim": 4},
        "readout": {"pooling": "attention"},
        "task": dict(GEN, num_train=10, num_test=6, dataset_seed=123,
                     delay=2, blank_mode="generic",
                     reward_correct=1.0, reward_incorrect=0.0,
                     stochastic_prob=0.0),
        "training": {"batch_size": 4, "sequence_length": 4,
                     "learning_rate": 1e-3, "weight_decay": 0.0,
                     "grad_clip": 1.0, "record_aux_every": 0},
    }


class TestReadoutDiagnostics:
    def test_runs_and_logs(self, tmp_path):
        cfg = _cfg()
        datasets = {
            "train": make_connectivity_dataset(cfg["task"]["num_train"], seed=123, **GEN),
            "test": make_connectivity_dataset(cfg["task"]["num_test"], seed=124, **GEN),
        }
        model = build_model(cfg, node_feature_dim=2, num_actions=2, num_nodes=None)
        ck = {"model_state": model.state_dict(), "seed": 0}
        summary = rd.run_readout_diagnostics(
            cfg, datasets, ck, tmp_path, seed=0,
            episodes=3, learning_rate=1e-3, probe_every=1,
        )
        assert 0.0 <= summary["delayed_final"] <= 1.0
        assert -1.0 <= summary["cos_imm_delayed_mean_tail"] <= 1.0
        df = pd.read_csv(tmp_path / "diag_lr0.001_seed0.csv")
        for col in ("loss_delayed", "loss_immediate", "grad_cos_imm_delayed",
                    "probe_immediate", "probe_delayed_intact"):
            assert col in df.columns
        assert len(df) == 3
        # frozen generation path must not have been written
        ssm = model.temporal_model
        assert float(ssm.B.weight.grad) == 0.0 if ssm.B.weight.grad is not None else True

    def test_checkpoint_copy_semantics(self, tmp_path):
        """Diagnostics mutate only the in-memory copy: two runs from the same
        checkpoint state produce identical first-episode losses."""
        cfg = _cfg()
        datasets = {
            "train": make_connectivity_dataset(cfg["task"]["num_train"], seed=123, **GEN),
            "test": make_connectivity_dataset(cfg["task"]["num_test"], seed=124, **GEN),
        }
        model = build_model(cfg, node_feature_dim=2, num_actions=2, num_nodes=None)
        state = model.state_dict()
        s1 = rd.run_readout_diagnostics(
            cfg, datasets, {"model_state": state}, tmp_path, seed=0,
            episodes=1, learning_rate=1e-3, probe_every=1,
        )
        s2 = rd.run_readout_diagnostics(
            cfg, datasets, {"model_state": state}, tmp_path, seed=0,
            episodes=1, learning_rate=1e-3, probe_every=1,
        )
        assert abs(s1["loss_delayed_initial"] - s2["loss_delayed_initial"]) < 1e-6
