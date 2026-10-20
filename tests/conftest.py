import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def tiny_config() -> dict:
    """Small-but-realistic config used across model tests."""
    return {
        "gnn": {
            "layers": 2,
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


@pytest.fixture()
def cfg() -> dict:
    return tiny_config()


@pytest.fixture()
def small_graph():
    """Ring graph with distinct node features, 8 nodes."""
    n, f = 8, 4
    torch.manual_seed(0)
    edges = [(i, (i + 1) % n) for i in range(n)]
    from artificial_system.data.schema import GraphData

    return GraphData(
        node_features=torch.randn(n, f),
        edge_index=torch.tensor(edges, dtype=torch.long).t(),
    )


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
