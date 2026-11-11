"""System-level query API (M1-b, TECHNICAL_REPORT §10)."""
import pytest
import torch

from artificial_system.data.graph_tasks import make_connectivity_dataset
from artificial_system.models.gnn_ssm import build_model
from artificial_system.inspection import SystemHandle
from artificial_system.utils.checkpoints import (
    save_checkpoint,
    save_execution_state,
)


def api_config() -> dict:
    return {
        "gnn": {"layers": 2, "hidden_dim": 16, "dropout": 0.0,
                "aggregation": "mean", "residual": True,
                "use_edge_features": False},
        "ssm": {"state_dim": 16, "diagonal": True, "max_radius": 0.99,
                "init_decay": 0.9, "nonlinearity": "tanh"},
        "readout": {"pooling": "mean", "head_hidden": 0},
        "task": {"num_actions": 2, "delay": 2, "marker_dim": 2},
        "training": {"batch_size": 4, "sequence_length": 8},
    }


@pytest.fixture(scope="module")
def dataset():
    return make_connectivity_dataset(
        num_instances=8, num_nodes=12, feature_dim=4, marker_dim=2,
        max_path_len=2, edge_density=0.2, seed=3,
    )


@pytest.fixture()
def exec_state_checkpoint(tmp_path, dataset):
    torch.manual_seed(0)
    model = build_model(api_config(), node_feature_dim=4, edge_feature_dim=0,
                        num_actions=2, num_nodes=dataset.graphs[0].num_nodes)
    path = save_execution_state(
        tmp_path / "exec_state.pt", model, global_step=24, seed=1,
        config=api_config(),
        extra={"model_dims": {"node_feature_dim": 4, "edge_feature_dim": 0,
                              "num_actions": 2,
                              "num_nodes": dataset.graphs[0].num_nodes}},
    )
    return path


class TestSystemHandle:
    def test_load_execution_state_and_summary(self, exec_state_checkpoint):
        handle = SystemHandle.load(exec_state_checkpoint)
        s = handle.summary()
        assert s["step"] == 24
        assert s["seed"] == 1
        assert s["num_parameters"] > 0
        assert "env_info" in s

    def test_probe_keys(self, exec_state_checkpoint, dataset):
        handle = SystemHandle.load(exec_state_checkpoint)
        probe = handle.probe(dataset, delay=0, batch_size=4, marker_dim=2)
        assert 0.0 <= probe["accuracy"] <= 1.0
        assert "accuracy_by_class" in probe

    def test_inspect_state_finite(self, exec_state_checkpoint, dataset):
        handle = SystemHandle.load(exec_state_checkpoint)
        out = handle.inspect_state(dataset, delay=2, batch_size=4, marker_dim=2)
        assert set(out["decode"]) == {"post_present", "pre_response"}
        for pt, dec in out["decode"].items():
            assert 0.0 <= dec["accuracy"] <= 1.0
            assert dec["n"] == len(dataset)
        assert all(v > 0.0 and v == v for v in out["state_norms"].values())
        assert 0.0 <= out["accuracy"] <= 1.0

    def test_legacy_checkpoint_requires_dims(self, tmp_path, dataset):
        torch.manual_seed(0)
        model = build_model(api_config(), node_feature_dim=4, edge_feature_dim=0,
                            num_actions=2, num_nodes=dataset.graphs[0].num_nodes)
        legacy = save_checkpoint(
            tmp_path / "legacy.pt", model, step=5, config=api_config(), seed=0,
        )
        with pytest.raises(ValueError):
            SystemHandle.load(legacy)
        handle = SystemHandle.load(legacy, node_feature_dim=4, edge_feature_dim=0,
                                   num_actions=2,
                                   num_nodes=dataset.graphs[0].num_nodes)
        assert handle.summary()["step"] == 5
