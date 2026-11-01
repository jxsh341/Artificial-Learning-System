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


class TestCaptureMode:
    def test_delayed_capture_stops_at_threshold(self, tmp_path):
        """The H7 capture stage halts as soon as the delayed probe crosses the
        threshold, and the summary records the capture event."""
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
            cfg, datasets, ck, "h7_capture_hold", tmp_path, seed=0,
            schedule_spec={"stage3": {"episodes": 6}, "stage4": {"episodes": 2},
                           "capture_probe_every": 2, "capture_threshold": 0.55},
        )
        assert summary["arm"] == "h7_capture_hold"
        assert (tmp_path / "arm_summary.json").exists()
        # capture either fired within 6 episodes or was reported uncaptured
        cap = summary["capture_info"]
        assert cap is None or (cap["stage"] == 1 and cap["episode_in_stage"] % 2 == 0)

    def test_run_episodes_capture_returns(self):
        """The capture executor returns (captured, used, probe) tuples."""
        model = _model()
        ds = make_connectivity_dataset(8, seed=11, **GEN)
        from artificial_system.environment.graph_environment import (
            GraphClassificationEnvironment, make_generic_graph,
        )
        task = dict(GEN, delay=2, blank_mode="generic",
                    reward_correct=1.0, reward_incorrect=0.0, stochastic_prob=0.0)
        env = GraphClassificationEnvironment(ds, task, seed=0)
        blank = make_generic_graph(GEN["num_nodes"], GEN["feature_dim"], 5)
        captured, used, cap = cr._run_episodes(
            (None, _trainer(model, env)),
            episodes=2, mode="delayed_capture", model=model, test_ds=ds,
            probe_kwargs={"delay": 2, "marker_dim": 2, "generic_blank": blank},
            capture_threshold=1.1,  # unreachable -> runs to completion
            probe_every=1,
        )
        assert captured is False and used == 2 and cap is None


def _trainer(model, env):
    from artificial_system.training.trainer import Trainer

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return Trainer(model, env, optimizer, {"batch_size": 4, "sequence_length": 4})


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


class TestDelayOverride:
    def test_alternating_events_mode_runs(self):
        """The event-balanced joint stage (§8m) pairs one delayed episode with
        one immediate step and reports the pair count."""
        model = _model()
        ds = make_connectivity_dataset(8, seed=11, **GEN)
        from artificial_system.environment.graph_environment import (
            GraphClassificationEnvironment,
        )
        task_del = dict(GEN, delay=2, blank_mode="generic",
                        reward_correct=1.0, reward_incorrect=0.0, stochastic_prob=0.0)
        task_imm = dict(task_del, delay=0)
        env_del = GraphClassificationEnvironment(ds, task_del, seed=0)
        env_imm = GraphClassificationEnvironment(ds, task_imm, seed=1)
        tr_del = _trainer(model, env_del)
        tr_step = _trainer(model, env_imm)
        tr_step.seq_len = 1  # one step = one immediate response event
        captured, used, cap = cr._run_episodes(
            (None, tr_del, tr_step), episodes=3, mode="alternating_events",
        )
        assert captured is False and used == 3 and cap is None

    def test_alternating_events_requires_step_trainer(self):
        import pytest
        model = _model()
        ds = make_connectivity_dataset(8, seed=11, **GEN)
        from artificial_system.environment.graph_environment import (
            GraphClassificationEnvironment,
        )
        env = GraphClassificationEnvironment(ds, dict(GEN, delay=2, blank_mode="generic"), seed=0)
        tr = _trainer(model, env)
        with pytest.raises(ValueError):
            cr._run_episodes((None, tr), episodes=1, mode="alternating_events")

    def test_success_rubric_matches_inline_version(self, tmp_path):
        """The factored rubric is the pre-registered one: a model that answers
        correctly but ignores donated state must fail all_criteria."""
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
            cfg, datasets, ck, "h7b_capture_alternate", tmp_path, seed=0,
            schedule_spec={"stage3": {"episodes": 2}, "stage4": {"episodes": 2},
                           "capture_probe_every": 1, "capture_threshold": 1.1},
        )
        # rubric produced by the factored helper, with the documented rule
        r = summary["rubric"]
        expected = all(r[k] for k in ("immediate_ok", "delayed_ok", "subst_same_ok",
                                      "zero_chance", "shuffled_chance")) \
            and r["subst_diff_follows"] >= 0.9 and r["decode_preserved"]
        assert r["all_criteria"] == expected
        assert set(r) == {"immediate_ok", "delayed_ok", "subst_same_ok",
                          "subst_diff_follows", "zero_chance", "shuffled_chance",
                          "decode_preserved", "state_norm_ratio_post_over_pre",
                          "all_criteria"}

    def test_delay_override_threads_through_without_mutating_cfg(self, tmp_path):
        """§8l stage B: the intervention must run at the overridden delay
        (training envs, capture probes, state summaries, final battery) while
        the caller's config keeps the checkpoint's original delay."""
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
            cfg, datasets, ck, "h7b_capture_alternate", tmp_path, seed=0,
            schedule_spec={"stage3": {"episodes": 2}, "stage4": {"episodes": 2},
                           "capture_probe_every": 1, "capture_threshold": 1.1},
            delay_override=5,
        )
        assert summary["delay"] == 5
        assert summary["env_delays"] == {"delayed": 5, "immediate": 0}
        assert summary["episodes_total"] == 4
        assert "decode_pre_response" in summary["pre_analysis"]
        # the source config is untouched: the checkpoint's delay stays 2
        assert cfg["task"]["delay"] == 2
