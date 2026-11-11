"""Execution-state checkpoint contract (M1-a, TECHNICAL_REPORT §10).

Covers: RNG capture/restore, environment stream serialization round-trips,
the full save/resume round-trip, episode-boundary resume exactness, and the
registered mid-episode semantics (episode restart from an episode-start
snapshot).
"""
import random

import numpy as np
import pytest
import torch

from artificial_system.data.graph_tasks import make_connectivity_dataset
from artificial_system.environment.cue_environment import CueLearningEnvironment
from artificial_system.environment.graph_environment import (
    GraphClassificationEnvironment,
)
from artificial_system.models.gnn_ssm import build_model
from artificial_system.training.trainer import Trainer
from artificial_system.utils.checkpoints import (
    capture_rng_states,
    resume_execution_state,
    restore_rng_states,
    save_execution_state,
)


def exec_config() -> dict:
    return {
        "gnn": {"layers": 2, "hidden_dim": 16, "dropout": 0.0,
                "aggregation": "mean", "residual": True,
                "use_edge_features": False},
        "ssm": {"state_dim": 16, "diagonal": True, "max_radius": 0.99,
                "init_decay": 0.9, "nonlinearity": "tanh"},
        "readout": {"pooling": "mean", "head_hidden": 0},
        "training": {"batch_size": 4, "sequence_length": 8, "grad_clip": 1.0},
    }


@pytest.fixture(scope="module")
def dataset():
    return make_connectivity_dataset(
        num_instances=12, num_nodes=10, feature_dim=4, marker_dim=2,
        max_path_len=2, edge_density=0.2, seed=0,
    )


def fresh_arm(seed: int, dataset):
    """A freshly seeded (model, optimizer, trainer, envs) triple."""
    torch.manual_seed(seed)
    model = build_model(exec_config(), node_feature_dim=4, edge_feature_dim=0,
                        num_actions=2, num_nodes=dataset.graphs[0].num_nodes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    torch.manual_seed(seed + 777)
    env_imm = GraphClassificationEnvironment(
        dataset, {**exec_config()["training"], "delay": 0, "marker_dim": 2,
                  "blank_mode": "generic"}, seed=seed * 100 + 1)
    env_del = GraphClassificationEnvironment(
        dataset, {**exec_config()["training"], "delay": 2, "marker_dim": 2,
                  "blank_mode": "generic"}, seed=seed * 100 + 2)
    envs = {"immediate": env_imm, "delayed": env_del}
    trainer_imm = Trainer(model, env_imm, optimizer, exec_config()["training"])
    trainer_del = Trainer(model, env_del, optimizer, exec_config()["training"])
    return model, optimizer, trainer_imm, trainer_del, envs


def interleaved(trainer_imm, trainer_del, episodes: int):
    done = 0
    while done < episodes:
        trainer_imm.train(1)
        done += 1
        if done < episodes:
            trainer_del.train(1)
            done += 1


class TestRngCaptureRestore:
    def test_torch_stream_restored(self):
        torch.manual_seed(42)
        torch.randn(3)  # consume
        captured = capture_rng_states()
        a = torch.randn(5)
        torch.randn(4)  # advance past the capture point
        restore_rng_states(captured)
        b = torch.randn(5)
        assert torch.equal(a, b)

    def test_numpy_and_python_restored(self):
        random.seed(7)
        np.random.seed(7)
        random.random()
        np.random.random(3)
        captured = capture_rng_states()
        pa, na = random.random(), np.random.random(2)
        random.random()
        np.random.random(4)
        restore_rng_states(captured)
        assert random.random() == pa
        assert np.array_equal(np.random.random(2), na)


class TestEnvStateRoundTrip:
    def test_graph_env_mid_trial(self, dataset):
        task = {"delay": 2, "marker_dim": 2, "blank_mode": "generic"}
        env = GraphClassificationEnvironment(dataset, task, seed=5)
        env.reset(4)
        env.step(torch.zeros(4, dtype=torch.long))  # mid-trial
        state = env.state_dict()

        clone = GraphClassificationEnvironment(dataset, task, seed=99)
        clone.load_state_dict(state)

        actions = torch.ones(4, dtype=torch.long)
        obs_a, r_a, _, info_a = env.step(actions)
        obs_b, r_b, _, info_b = clone.step(actions)
        assert torch.equal(obs_a.node_features, obs_b.node_features)
        assert torch.equal(r_a, r_b)
        assert torch.equal(info_a["targets"], info_b["targets"])

    def test_cue_env_preserves_mapping(self):
        base = None
        torch.manual_seed(0)
        n, f = 8, 4
        from artificial_system.data.schema import GraphData

        base = GraphData(
            node_features=torch.randn(n, f),
            edge_index=torch.tensor([(i, (i + 1) % n) for i in range(n)],
                                    dtype=torch.long).t(),
        )
        task = {"delay": 1, "cue_feature_dim": 4, "num_cues": 4}
        env = CueLearningEnvironment(base, task, seed=3)
        env.reset(4)
        env.reverse()  # reversal applied before the save must survive
        env.step(torch.zeros(4, dtype=torch.long))
        state = env.state_dict()

        clone = CueLearningEnvironment(base, task, seed=99)
        clone.load_state_dict(state)
        assert torch.equal(clone.mapping_tensor(), env.mapping_tensor())

        obs_a = env.reset(4)
        obs_b = clone.reset(4)
        assert torch.equal(obs_a.node_features, obs_b.node_features)
        assert torch.equal(env.current_targets(), clone.current_targets())


class TestExecutionState:
    def test_round_trip_restores_params_env_and_rng(self, dataset, tmp_path):
        model, optimizer, trainer_imm, trainer_del, envs = fresh_arm(0, dataset)
        interleaved(trainer_imm, trainer_del, 3)
        expected_step = trainer_imm.execution_state()["global_step"]
        path = save_execution_state(
            tmp_path / "exec_state.pt", model, optimizer,
            global_step=expected_step,
            envs=envs, config=exec_config(), seed=0,
            extra={"model_dims": {"node_feature_dim": 4, "edge_feature_dim": 0,
                                  "num_actions": 2,
                                  "num_nodes": dataset.graphs[0].num_nodes}},
        )

        model2, optimizer2, t2_imm, t2_del, envs2 = fresh_arm(123, dataset)
        payload = resume_execution_state(path, model2, optimizer2, envs=envs2)
        assert payload["global_step"] == expected_step
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)
        # environment streams at identical positions
        assert torch.equal(envs["immediate"]._idxs, envs2["immediate"]._idxs)
        assert torch.equal(envs["delayed"]._idxs, envs2["delayed"]._idxs)

    def test_episode_boundary_resume_is_bit_exact(self, dataset, tmp_path):
        model, _, t_imm, t_del, _ = fresh_arm(0, dataset)
        interleaved(t_imm, t_del, 4)
        control_params = [p.detach().clone() for p in model.parameters()]

        model2, optimizer2, u_imm, u_del, envs2 = fresh_arm(0, dataset)
        interleaved(u_imm, u_del, 2)
        steps_imm = u_imm.execution_state()
        steps_del = u_del.execution_state()
        save_execution_state(tmp_path / "mid.pt", model2, optimizer2,
                             global_step=steps_imm["global_step"],
                             envs=envs2, config=exec_config(), seed=0)
        resume_execution_state(tmp_path / "mid.pt", model2, optimizer2, envs=envs2)
        u_imm.load_execution_state(steps_imm)
        u_del.load_execution_state(steps_del)
        interleaved(u_imm, u_del, 2)

        for p_c, p_t in zip(control_params, model2.parameters()):
            assert torch.equal(p_c, p_t)

    def test_mid_episode_pause_replays_from_episode_start(self, dataset, tmp_path):
        """Registered E1b semantics: the interrupt kills the in-flight episode;
        resume restores the snapshot taken at that episode's reset and replays
        it — final parameters match the uninterrupted run bit-for-bit."""
        model, _, t_imm, t_del, _ = fresh_arm(0, dataset)
        interleaved(t_imm, t_del, 4)
        control_params = [p.detach().clone() for p in model.parameters()]

        model2, optimizer2, u_imm, u_del, envs2 = fresh_arm(0, dataset)
        seen = {"episodes": 0}
        snap = tmp_path / "episode_start.pt"

        def hook(episode, trainer):  # fires BEFORE env.reset of each episode
            if seen["episodes"] == 2:  # the episode to be interrupted (0-based)
                save_execution_state(snap, trainer.model, trainer.optimizer,
                                     global_step=trainer._current_step,
                                     envs=envs2, config=exec_config(), seed=0)
            seen["episodes"] += 1

        u_imm.episode_snapshot_fn = hook
        u_del.episode_snapshot_fn = hook
        interleaved(u_imm, u_del, 2)          # episodes 0, 1 complete
        steps_imm = u_imm.execution_state()
        steps_del = u_del.execution_state()
        u_imm.train(1, pause_after_steps=3)   # episode 2 interrupted mid-flight
        assert seen["episodes"] == 3

        resume_execution_state(snap, model2, optimizer2, envs=envs2)
        u_imm.load_execution_state(steps_imm)
        u_del.load_execution_state(steps_del)
        u_imm.episode_snapshot_fn = None
        u_del.episode_snapshot_fn = None
        interleaved(u_imm, u_del, 2)          # replay episode 2 + episode 3

        for p_c, p_t in zip(control_params, model2.parameters()):
            assert torch.equal(p_c, p_t)

    def test_pause_disabled_is_noop(self, dataset):
        model, _, t_imm, t_del, _ = fresh_arm(0, dataset)
        rec_a = t_imm.train(2)
        rec_b = t_imm.train(2, pause_after_steps=None)
        assert len(rec_a) == 2 * 8 and len(rec_b) == 2 * 8
