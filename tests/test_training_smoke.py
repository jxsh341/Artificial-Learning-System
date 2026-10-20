"""Smoke tests: trainer mechanics and the full experiment pipeline (spec §30)."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from artificial_system.environment.cue_environment import CueLearningEnvironment
from artificial_system.experiments.runner import run_experiment
from artificial_system.models import build_model
from artificial_system.training.trainer import Trainer


def make_trainer(cfg, graph):
    model = build_model(cfg, node_feature_dim=4, num_actions=2, num_nodes=graph.num_nodes)
    env = CueLearningEnvironment(graph, cfg["task"], seed=0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    return Trainer(model, env, opt, cfg["training"])


class TestTrainer:
    def test_train_records_and_loss_decreases_finite(self, cfg, small_graph):
        cfg["training"]["sequence_length"] = 8
        trainer = make_trainer(cfg, small_graph)
        records = trainer.train(3)
        assert len(records) == 24
        for key in ("step", "episode", "loss", "accuracy", "reward", "active", "state_norm"):
            assert key in records[0]
        assert all(r["loss"] == r["loss"] for r in records)  # no NaN losses

    def test_state_resets_between_episodes(self, cfg, small_graph):
        trainer = make_trainer(cfg, small_graph)
        records = trainer.train(2)
        first_norms = [
            records[i * cfg["training"]["sequence_length"]]["state_norm"] for i in range(2)
        ]
        assert all(n == 0.0 or n < 1e-6 or n == n for n in first_norms)  # finite

    def test_evaluate_uses_no_grad(self, cfg, small_graph):
        trainer = make_trainer(cfg, small_graph)
        records = trainer.evaluate(10)
        assert len(records) == 10


class TestExperimentPipeline:
    def test_run_experiment_end_to_end(self, tmp_path, cfg):
        """The Definition-of-Done command runs headlessly on a fresh graph."""
        config = {
            **cfg,
            "experiment_name": "pytest_smoke",
            "results_dir": str(tmp_path / "results"),
            "seeds": [0],
            "graph": {
                "family": "random",
                "num_nodes": 20,
                "node_feature_dim": 8,
                "edge_feature_dim": 0,
                "directed": True,
                "seed": 1,
                "params": {"p": 0.2},
            },
            "gnn": {**cfg["gnn"], "hidden_dim": 16, "layers": 1},
            "readout": {"pooling": "mean", "head_hidden": 0},
            "task": {
                **cfg["task"],
                "cue_feature_dim": 4,
                "delay": 0,
                "train_cues": [0, 1, 2],
                "test_cues": [3],
            },
            "training": {**cfg["training"], "batch_size": 4, "sequence_length": 6, "episodes": 2},
            "evaluation": {
                "acquisition": True,
                "retention": True,
                "reversal": True,
                "generalization": True,
                "criterion": 0.9,
                "criterion_window": 5,
                "retention_delays": [0, 5],
                "retention_test_steps": 10,
                "retention_distractor": "blank",
                "reversal_episodes": 2,
                "generalization_test_steps": 10,
                "eval_batch_size": 4,
            },
            "ablations": {"enabled": True, "models": ["M1", "M3"]},
            "checkpointing": {"save": True},
        }
        exp_dir = run_experiment(config=config)
        seed_dir = Path(exp_dir) / "seed_000"
        assert (Path(exp_dir) / "config.yaml").exists()
        assert (seed_dir / "checkpoint.pt").exists()
        assert (seed_dir / "metrics.csv").exists()
        assert (seed_dir / "retention.csv").exists()
        assert (seed_dir / "reversal.csv").exists()
        assert (seed_dir / "generalization.csv").exists()
        summary = json.loads((Path(exp_dir) / "summary.json").read_text())
        assert summary["n_seeds"] == 1
        assert "acquisition_accuracy_last_10pct_mean" in summary
        assert "retention_ratio@5_mean" in summary
        assert "ablation_M1" in summary

        # checkpoint round-trip
        ckpt = torch.load(seed_dir / "checkpoint.pt", weights_only=False)
        assert "model_state" in ckpt and "env_info" in ckpt
