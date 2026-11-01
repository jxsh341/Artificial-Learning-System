"""Unit tests for the readout–probe geometry (§8l stage A)."""
import numpy as np
import torch

from artificial_system.environment.graph_environment import make_generic_graph
from artificial_system.evaluation import readout_geometry as rg
from artificial_system.models.gnn_ssm import build_model


def _cfg():
    return {
        "graph": {"node_feature_dim": 2},
        "gnn": {"layers": 2, "hidden_dim": 8, "aggregation": "sum"},
        "ssm": {"state_dim": 4},
        "readout": {"pooling": "attention"},
        "task": {"num_nodes": 20, "marker_dim": 2, "delay": 2,
                 "blank_mode": "generic", "edge_density": 0.2},
    }


class TestReadoutGeometry:
    def test_effective_direction_matches_finite_difference(self):
        """The local gradient must equal the finite-difference derivative of
        the actual logit margin — the basis of the alignment measure.  (The
        state→logit map is *not* globally linear because of the SSM tanh.)"""
        cfg = _cfg()
        model = build_model(cfg, node_feature_dim=2, num_actions=2, num_nodes=None)
        model.eval()
        blank = make_generic_graph(20, 2, seed=0)
        d = int(cfg["ssm"]["state_dim"])

        s = torch.randn(1, d)
        w_r = rg.readout_direction(model, blank, s)
        for _ in range(3):
            v = torch.randn(d)
            eps = 1e-3
            fd = (rg.margin_at(model, blank, s + eps * v)
                  - rg.margin_at(model, blank, s - eps * v)) / (2 * eps)
            assert abs(fd - float(w_r @ v)) < 1e-3

    def test_boundary_statistics_separate_translation_from_alignment(self):
        labels = np.array([0, 0, 1, 1])

        # boundary midway between the class means
        aligned = rg.boundary_statistics(np.array([-0.5, -0.4, 0.4, 0.5]), labels)
        assert abs(aligned["boundary_vs_mid"]) <= 0.5
        assert aligned["class_margin_separation"] > 0

        # both class means on the positive side: class 0 entirely misclassified
        translated = rg.boundary_statistics(np.array([0.5, 0.6, 1.0, 1.1]), labels)
        assert translated["boundary_vs_mid"] < -0.5
        assert translated["class_margin_separation"] > 0

        # and the mirror case: class 1 entirely misclassified
        mirrored = rg.boundary_statistics(np.array([-1.1, -1.0, -0.6, -0.5]), labels)
        assert mirrored["boundary_vs_mid"] > 0.5
