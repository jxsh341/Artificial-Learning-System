"""Live-session inspection (REQ-ALS-M1-1, ALS-M1-LIVE-INSPECT/1.0).

Validates the observational guarantee: capture + all queries leave the live
session bit-identical (parameters, optimizer, step counters, environment
streams, RNG states), do not advance execution, and do not change subsequent
training behaviour. Exact (bit/hash) comparisons throughout — every quantity
inspected here is deterministic on CPU, so exact equality is appropriate.
"""
import copy

import pytest
import torch

from artificial_system.data.graph_tasks import make_connectivity_dataset
from artificial_system.environment.graph_environment import (
    GraphClassificationEnvironment,
)
from artificial_system.inspection import (
    CONTRACT_VERSION,
    LiveSessionHandle,
    SystemHandle,
    parameter_digest,
)
from artificial_system.models.gnn_ssm import build_model
from artificial_system.training.trainer import Trainer
from artificial_system.utils.checkpoints import (
    resume_execution_state,
    save_execution_state,
)


def live_config() -> dict:
    return {
        "gnn": {"layers": 2, "hidden_dim": 16, "dropout": 0.0,
                "aggregation": "mean", "residual": True,
                "use_edge_features": False},
        "ssm": {"state_dim": 16, "diagonal": True, "max_radius": 0.99,
                "init_decay": 0.9, "nonlinearity": "tanh"},
        "readout": {"pooling": "mean", "head_hidden": 0},
        "training": {"batch_size": 4, "sequence_length": 8},
    }


@pytest.fixture(scope="module")
def dataset():
    return make_connectivity_dataset(
        num_instances=12, num_nodes=10, feature_dim=4, marker_dim=2,
        max_path_len=2, edge_density=0.2, seed=0,
    )


def fresh_session(seed: int, dataset):
    torch.manual_seed(seed)
    model = build_model(live_config(), node_feature_dim=4, edge_feature_dim=0,
                        num_actions=2, num_nodes=dataset.graphs[0].num_nodes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    task = {"delay": 2, "marker_dim": 2, "blank_mode": "generic"}
    env_imm = GraphClassificationEnvironment(
        dataset, {**task, "delay": 0}, seed=seed * 100 + 1)
    env_del = GraphClassificationEnvironment(dataset, task, seed=seed * 100 + 2)
    envs = {"immediate": env_imm, "delayed": env_del}
    cfg = live_config()["training"]
    t_imm = Trainer(model, env_imm, optimizer, cfg)
    t_del = Trainer(model, env_del, optimizer, cfg)
    return model, optimizer, t_imm, t_del, envs


def interleaved(t_imm, t_del, episodes: int):
    done = 0
    while done < episodes:
        t_imm.train(1)
        done += 1
        if done < episodes:
            t_del.train(1)
            done += 1


def session_state(model, optimizer, t_imm, t_del, envs):
    """Everything inspection must not touch (for exact comparisons)."""
    return {
        "param_digest": parameter_digest(model),
        "optimizer": optimizer.state_dict(),
        "step_imm": t_imm.execution_state()["global_step"],
        "step_del": t_del.execution_state()["global_step"],
        "env_imm": envs["immediate"].state_dict(),
        "env_del": envs["delayed"].state_dict(),
        "rng_torch": torch.get_rng_state(),
    }


def assert_state_equal(a, b):
    assert a["param_digest"] == b["param_digest"]
    assert a["step_imm"] == b["step_imm"] and a["step_del"] == b["step_del"]
    assert torch.equal(a["rng_torch"], b["rng_torch"])
    for key in ("env_imm", "env_del"):
        x, y = a[key], b[key]
        for k in x:
            same = (
                torch.equal(x[k], y[k]) if torch.is_tensor(x[k]) else x[k] == y[k]
            )
            assert same, f"session state {key}/{k} changed"
    assert str(a["optimizer"]) == str(b["optimizer"])


def full_query_battery(handle, dataset):
    """Every delivered operation, once, through the uniform dispatcher."""
    return [
        handle.query("summary"),
        handle.query("probe", dataset=dataset, delay=0, batch_size=4,
                     marker_dim=2),
        handle.query("probe", dataset=dataset, delay=2, batch_size=4,
                     marker_dim=2, generic_blank=None),
        handle.query("inspect_state", dataset=dataset, delay=2, batch_size=4,
                     marker_dim=2),
        handle.query("state_digest"),
        handle.query("availability"),
    ]


class TestCaptureReturnsActiveState:
    def test_capture_reflects_live_session(self, dataset):
        model, optimizer, t_imm, t_del, envs = fresh_session(0, dataset)
        interleaved(t_imm, t_del, 5)
        handle = LiveSessionHandle.capture(
            model, optimizer=optimizer, trainers={"immediate": t_imm,
                                                  "delayed": t_del},
            envs=envs, session_id="session-A")
        s = handle.summary()
        assert s["supported"] is True
        assert s["session_id"] == "session-A"
        assert s["steps"] == {"immediate": 24, "delayed": 16}  # 3/2 eps x 8 steps
        assert s["step"] == 40
        # the frozen copy is bit-identical to the live model at capture
        for p_live, p_frozen in zip(model.parameters(), handle.model.parameters()):
            assert torch.equal(p_live, p_frozen)
        assert s["captured_digest"] == parameter_digest(model)
        assert s["num_parameters"] > 0
        assert s["optimizer"]["present"] is True
        assert s["envs"]["immediate"]["available"] is True

    def test_availability_reports_what_was_captured(self, dataset):
        model, optimizer, t_imm, t_del, envs = fresh_session(1, dataset)
        interleaved(t_imm, t_del, 2)
        handle = LiveSessionHandle.capture(model, trainers={"i": t_imm})
        av = handle.query("availability")
        assert av["supported"] and av["model"] is True
        assert av["optimizer"] is False          # not captured
        assert av["trainer_step"] is True
        assert av["envs"] is None                # not captured
        full = LiveSessionHandle.capture(
            model, optimizer=optimizer,
            trainers={"i": t_imm, "d": t_del}, envs=envs)
        assert full.query("availability")["optimizer"] is True


class TestReadOnlyGuarantee:
    def test_inspection_does_not_advance_or_mutate(self, dataset):
        model, optimizer, t_imm, t_del, envs = fresh_session(2, dataset)
        interleaved(t_imm, t_del, 4)
        before = session_state(model, optimizer, t_imm, t_del, envs)
        handle = LiveSessionHandle.capture(
            model, optimizer=optimizer,
            trainers={"immediate": t_imm, "delayed": t_del}, envs=envs)
        responses = full_query_battery(handle, dataset)
        assert all(r["supported"] for r in responses)
        after = session_state(model, optimizer, t_imm, t_del, envs)
        assert_state_equal(before, after)  # params, optimizer, steps, envs, RNG

    def test_model_parameters_bit_identical(self, dataset):
        model, optimizer, t_imm, t_del, envs = fresh_session(3, dataset)
        interleaved(t_imm, t_del, 3)
        live_digest = parameter_digest(model)
        handle = LiveSessionHandle.capture(
            model, trainers={"i": t_imm, "d": t_del})
        full_query_battery(handle, dataset)
        assert parameter_digest(model) == live_digest

    def test_optimizer_state_unchanged(self, dataset):
        model, optimizer, t_imm, t_del, envs = fresh_session(4, dataset)
        interleaved(t_imm, t_del, 3)
        opt_before = copy.deepcopy(optimizer.state_dict())
        handle = LiveSessionHandle.capture(
            model, optimizer=optimizer, trainers={"i": t_imm, "d": t_del})
        full_query_battery(handle, dataset)
        opt_after = optimizer.state_dict()
        assert str(opt_after) == str(opt_before)

    def test_live_module_training_flag_untouched(self, dataset):
        """The existing measurement helpers flip module .train()/.eval(); the
        live handle must not (the frozen copy is switched instead)."""
        model, optimizer, t_imm, t_del, envs = fresh_session(5, dataset)
        interleaved(t_imm, t_del, 2)
        assert model.training is True  # trainers leave it in train mode
        handle = LiveSessionHandle.capture(model, trainers={"i": t_imm})
        handle.query("probe", dataset=dataset, batch_size=4)
        handle.query("inspect_state", dataset=dataset, batch_size=4)
        assert model.training is True          # live module untouched
        assert handle.model is not model       # queries ran on a detached copy

    def test_subsequent_behaviour_bit_identical(self, dataset):
        """Continue training after a full inspection battery; the session must
        match an never-inspected twin bit-for-bit."""
        control_model, _, c_imm, c_del, _ = fresh_session(6, dataset)
        interleaved(c_imm, c_del, 8)
        control_digest = parameter_digest(control_model)

        model, optimizer, t_imm, t_del, envs = fresh_session(6, dataset)
        interleaved(t_imm, t_del, 4)
        handle = LiveSessionHandle.capture(
            model, optimizer=optimizer,
            trainers={"immediate": t_imm, "delayed": t_del}, envs=envs)
        full_query_battery(handle, dataset)
        interleaved(t_imm, t_del, 4)

        assert parameter_digest(model) == control_digest

    def test_handle_frozen_while_live_advances(self, dataset):
        model, optimizer, t_imm, t_del, envs = fresh_session(7, dataset)
        interleaved(t_imm, t_del, 4)
        handle = LiveSessionHandle.capture(
            model, trainers={"immediate": t_imm, "delayed": t_del})
        frozen_probe = handle.query("probe", dataset=dataset, batch_size=4)
        interleaved(t_imm, t_del, 4)  # live session moves on
        assert handle.query("summary")["steps"] == {"immediate": 16,
                                                    "delayed": 16}
        assert handle.query("probe", dataset=dataset, batch_size=4) == frozen_probe
        assert parameter_digest(model) != handle.summary()["captured_digest"]


class TestConsistencyAndUnsupported:
    def test_repeated_inspection_identical(self, dataset):
        model, optimizer, t_imm, t_del, envs = fresh_session(8, dataset)
        interleaved(t_imm, t_del, 4)
        handle = LiveSessionHandle.capture(
            model, trainers={"i": t_imm, "d": t_del})
        p1 = handle.query("probe", dataset=dataset, delay=2, batch_size=4)
        p2 = handle.query("probe", dataset=dataset, delay=2, batch_size=4)
        assert p1 == p2
        assert handle.query("state_digest") == handle.query("state_digest")
        s1 = handle.query("inspect_state", dataset=dataset, delay=2, batch_size=4)
        s2 = handle.query("inspect_state", dataset=dataset, delay=2, batch_size=4)
        for pt in ("post_present", "delay_1", "pre_response"):
            assert torch.equal(s1["states"][pt], s2["states"][pt])
        assert s1["decode"] == s2["decode"] and s1["accuracy"] == s2["accuracy"]

    def test_unsupported_queries_explicit(self, dataset):
        model, _, t_imm, t_del, _ = fresh_session(9, dataset)
        handle = LiveSessionHandle.capture(model, trainers={"i": t_imm})
        r = handle.query("predict_next_episode")
        assert r["supported"] is False
        assert "unknown query" in r["reason"]
        assert r["contract_version"] == CONTRACT_VERSION
        r2 = handle.query("probe")  # dataset missing
        assert r2["supported"] is False and "dataset" in r2["reason"]
        r3 = handle.query("inspect_state")
        assert r3["supported"] is False


class TestExecutionBoundaries:
    def test_capture_at_episode_boundary_midrun(self, dataset):
        """Boundary: between two trainer calls. Inspection must not disturb
        the interleaved schedule that follows."""
        control_model, _, c_imm, c_del, _ = fresh_session(10, dataset)
        interleaved(c_imm, c_del, 8)
        control_digest = parameter_digest(control_model)

        model, optimizer, t_imm, t_del, envs = fresh_session(10, dataset)
        interleaved(t_imm, t_del, 4)          # boundary: episode 4 just closed
        LiveSessionHandle.capture(
            model, optimizer=optimizer,
            trainers={"immediate": t_imm, "delayed": t_del},
            envs=envs).query("summary")
        interleaved(t_imm, t_del, 4)
        assert parameter_digest(model) == control_digest

    def test_capture_at_mid_episode_pause_boundary(self, dataset, tmp_path):
        """Boundary: the registered M1-a mid-episode pause. Inspection at the
        pause must not corrupt the episode-start snapshot / replay machinery."""
        control_model, _, c_imm, c_del, _ = fresh_session(11, dataset)
        interleaved(c_imm, c_del, 6)
        control_digest = parameter_digest(control_model)

        model, optimizer, u_imm, u_del, envs = fresh_session(11, dataset)
        snap = tmp_path / "episode_start.pt"
        seen = {"n": 0}

        def hook(episode, trainer):
            if seen["n"] == 2:
                save_execution_state(snap, trainer.model, trainer.optimizer,
                                     global_step=trainer._current_step,
                                     envs=envs, config=live_config(), seed=11)
            seen["n"] += 1

        u_imm.episode_snapshot_fn = hook
        u_del.episode_snapshot_fn = hook
        interleaved(u_imm, u_del, 2)
        u_imm.train(1, pause_after_steps=3)   # supported mid-episode boundary
        assert seen["n"] == 3

        handle = LiveSessionHandle.capture(
            model, optimizer=optimizer,
            trainers={"immediate": u_imm, "delayed": u_del}, envs=envs)
        full_query_battery(handle, dataset)

        resume_execution_state(snap, model, optimizer, envs=envs)
        u_imm.load_execution_state({"global_step": 8})
        u_del.load_execution_state({"global_step": 8})
        u_imm.episode_snapshot_fn = None
        u_del.episode_snapshot_fn = None
        interleaved(u_imm, u_del, 4)          # replay + remaining episodes
        assert parameter_digest(model) == control_digest


class TestModeSeparationAndCompat:
    def test_live_and_checkpoint_modes_agree(self, dataset, tmp_path):
        """The same state inspected live and via an execution-state checkpoint
        yields identical measurements; the two modes are correctly tagged."""
        model, optimizer, t_imm, t_del, envs = fresh_session(12, dataset)
        interleaved(t_imm, t_del, 4)
        live = LiveSessionHandle.capture(
            model, trainers={"i": t_imm, "d": t_del})
        path = save_execution_state(
            tmp_path / "exec.pt", model, optimizer,
            global_step=t_imm.execution_state()["global_step"], envs=envs,
            config=live_config(), seed=12,
            extra={"model_dims": {"node_feature_dim": 4, "edge_feature_dim": 0,
                                  "num_actions": 2,
                                  "num_nodes": dataset.graphs[0].num_nodes}})
        ckpt = SystemHandle.load(path)
        lp = live.query("probe", dataset=dataset, delay=2, batch_size=4)
        cp = ckpt.probe(dataset, delay=2, batch_size=4)
        assert lp["accuracy"] == cp["accuracy"]
        assert lp["mode"] == "live_snapshot" and ckpt.summary()["mode"] == "checkpoint"

    def test_contract_version_and_modes(self, dataset):
        model, _, t_imm, _, _ = fresh_session(13, dataset)
        handle = LiveSessionHandle.capture(model, trainers={"i": t_imm})
        assert handle.summary()["contract_version"] == CONTRACT_VERSION
        assert handle.summary()["mode"] == "live_snapshot"
        assert handle.query("availability")["contract_version"] == CONTRACT_VERSION

    def test_existing_systemhandle_flows_preserved(self, dataset, tmp_path):
        """Backward compatibility: the checkpoint-backed API is unchanged."""
        model, _, t_imm, _, _ = fresh_session(14, dataset)
        path = save_execution_state(
            tmp_path / "exec.pt", model,
            global_step=8, config=live_config(), seed=14,
            extra={"model_dims": {"node_feature_dim": 4, "edge_feature_dim": 0,
                                  "num_actions": 2,
                                  "num_nodes": dataset.graphs[0].num_nodes}})
        handle = SystemHandle.load(path)
        s = handle.summary()
        assert s["step"] == 8 and s["mode"] == "checkpoint"
        probe = handle.probe(dataset, delay=0, batch_size=4, marker_dim=2)
        assert 0.0 <= probe["accuracy"] <= 1.0
        out = handle.inspect_state(dataset, delay=2, batch_size=4, marker_dim=2)
        assert set(out["decode"]) == {"post_present", "pre_response"}
