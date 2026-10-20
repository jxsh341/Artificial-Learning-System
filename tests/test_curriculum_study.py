"""Phase C/D tests: intact-vs-reset probe, computed-answer retention,
counterfactual probe, curriculum-study pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from artificial_system.data.graph_tasks import make_connectivity_dataset
from artificial_system.environment.graph_environment import GraphClassificationEnvironment
from artificial_system.evaluation.graph_classification import (
    probe_delayed_intact_vs_reset,
    run_counterfactual_probe,
    run_delay_retention,
)
from artificial_system.experiments.runner import run_curriculum_study
from artificial_system.models import build_model


def config(**task_overrides) -> dict:
    base = {
        "graph": {"node_feature_dim": 8, "edge_feature_dim": 0},
        "gnn": {
            "layers": 2,
            "hidden_dim": 16,
            "dropout": 0.0,
            "aggregation": "sum",
            "residual": True,
            "use_edge_features": False,
        },
        "ssm": {
            "state_dim": 8,
            "diagonal": True,
            "max_radius": 0.99,
            "init_decay": 0.9,
            "nonlinearity": "tanh",
        },
        "readout": {"pooling": "mean", "head_hidden": 0},
        "task": {
            "type": "graph_classification",
            "label_mode": "adjacency",
            "neg_min_distance": 3,
            "num_nodes": 30,
            "max_path_len": 3,
            "edge_density": 0.08,
            "marker_dim": 2,
            "directed": False,
            "num_train": 40,
            "num_test": 20,
            "delay": 2,
            "blank_mode": "generic",
            "reward_correct": 1.0,
            "reward_incorrect": 0.0,
            "stochastic_prob": 0.0,
        },
        "training": {"batch_size": 4, "sequence_length": 8, "grad_clip": 1.0},
    }
    base["task"].update(task_overrides)
    return base


@pytest.fixture(scope="module")
def dataset():
    return make_connectivity_dataset(
        40, num_nodes=30, feature_dim=8, marker_dim=2,
        edge_density=0.08, seed=0, label_mode="adjacency", neg_min_distance=3,
    )


@pytest.fixture(scope="module")
def generic_blank(dataset):
    env = GraphClassificationEnvironment(dataset, config()["task"], seed=0)
    env.reset(4)
    return env._generic_graph


class TestIntactVsReset:
    def test_keys_and_finite(self, dataset, generic_blank):
        model = build_model(config(), node_feature_dim=8, num_actions=2, num_nodes=None)
        out = probe_delayed_intact_vs_reset(
            model, dataset, delay=2, batch_size=8, generic_blank=generic_blank
        )
        assert set(out) == {"delayed_intact", "delayed_state_reset"}
        assert all(0.0 <= v <= 1.0 for v in out.values())


class TestDelayRetention:
    def test_rows_match_delays(self, dataset, generic_blank):
        model = build_model(config(), node_feature_dim=8, num_actions=2, num_nodes=None)
        rows, summary = run_delay_retention(
            model, dataset, delays=[0, 2, 5], batch_size=8, generic_blank=generic_blank
        )
        assert [r["delay"] for r in rows] == [0, 2, 5]
        assert summary["answer_retention@0"] == rows[0]["accuracy"]
        assert all(0.0 <= r["accuracy"] <= 1.0 for r in rows)


class TestCounterfactual:
    def test_conditions_and_pairing(self, dataset, generic_blank):
        model = build_model(config(), node_feature_dim=8, num_actions=2, num_nodes=None)
        rows, summary = run_counterfactual_probe(
            model, dataset, delay=2, batch_size=8, generic_blank=generic_blank, seed=0
        )
        conditions = {r["condition"] for r in rows}
        assert conditions == {
            "no_change", "update", "irrelevant",
            "update_label_flipped", "update_label_same",
        }
        assert all(0.0 <= v <= 1.0 for v in summary.values())
        # pairing excludes self: flipped + same-label coverage should cover the batch
        assert summary["counterfactual_no_change_accuracy"] == rows[0]["accuracy"]

    def test_deterministic_with_seed(self, dataset, generic_blank):
        model = build_model(config(), node_feature_dim=8, num_actions=2, num_nodes=None)
        _, s1 = run_counterfactual_probe(model, dataset, delay=2, batch_size=8,
                                         generic_blank=generic_blank, seed=5)
        _, s2 = run_counterfactual_probe(model, dataset, delay=2, batch_size=8,
                                         generic_blank=generic_blank, seed=5)
        assert s1 == s2


class TestCurriculumStudyPipeline:
    def test_study_end_to_end_smoke(self, tmp_path):
        """Tiny budget-matched study (conditions A + C) runs and exports results."""
        cfg = config(num_train=40, num_test=20)
        cfg.update({
            "experiment_name": "pytest_curr_study",
            "results_dir": str(tmp_path / "results"),
            "seeds": [0],
            "training": {**cfg["training"], "batch_size": 4, "sequence_length": 8},
            "study": {
                "episodes_total": 8,
                "chunk_size": 4,
                "success_threshold": 0.9,
                "delay_retention_delays": [0, 2],
                "counterfactual": True,
                "rewired_levels": [1.0],
                "conditions": {
                    "A_immediate": {"schedule": [["immediate", 8]]},
                    "C_staged": {"schedule": [["immediate", 4], ["delayed", 4]]},
                },
            },
            "checkpointing": {"save": True},
        })
        exp_dir = run_curriculum_study(config=cfg)
        cond_dir = Path(exp_dir) / "curriculum" / "A_immediate" / "seed_000"
        assert (cond_dir / "checkpoint.pt").exists()
        assert (cond_dir / "training_probes.csv").exists()
        assert (cond_dir / "counterfactual.csv").exists()
        assert (cond_dir / "delay_retention.csv").exists()
        summary = json.loads((Path(exp_dir) / "summary.json").read_text())
        assert set(summary) == {"A_immediate", "C_staged"}
        for cond in summary.values():
            assert "success_rate" in cond
            assert cond["per_seed"][0]["episodes_total"] == 8
        # budget matched: both conditions trained the same number of episodes
        assert summary["A_immediate"]["per_seed"][0]["episodes_total"] == \
            summary["C_staged"]["per_seed"][0]["episodes_total"]
