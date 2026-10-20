"""Environment tests (spec §26 + §12): cues, rewards, delays, reversal."""
from __future__ import annotations

import torch

from artificial_system.environment.cue_environment import IGNORE_INDEX, CueLearningEnvironment
from artificial_system.environment.tasks import make_cue_features, make_mapping


def make_env(cfg, graph, **overrides):
    task = {**cfg["task"], **overrides}
    return CueLearningEnvironment(graph, task, seed=0)


class TestBasics:
    def test_reset_returns_batched_graph(self, cfg, small_graph):
        env = make_env(cfg, small_graph, delay=0)
        obs = env.reset(4)
        assert obs.num_graphs == 4
        assert obs.num_nodes == 4 * small_graph.num_nodes
        assert obs.num_edges == 4 * small_graph.num_edges

    def test_cue_written_to_sensory_slice(self, cfg, small_graph):
        env = make_env(cfg, small_graph, delay=0)
        env.reset(2)
        d = env.cue_dim
        obs = env.cue_observation(torch.tensor([0, 3]))
        f0 = obs.node_features[: small_graph.num_nodes, :d]
        assert torch.allclose(f0, env.cue_features[0].expand_as(f0))
        f1 = obs.node_features[small_graph.num_nodes :, :d]
        assert torch.allclose(f1, env.cue_features[3].expand_as(f1))

    def test_correct_action_reward_inactive_step_zero(self, cfg, small_graph):
        env = make_env(cfg, small_graph, delay=0, reward_correct=1.0)
        env.reset(4)
        targets = env.current_targets()
        obs, rewards, done, info = env.step(targets.clone())
        assert torch.all(rewards == 1.0)

    def test_wrong_action_reward(self, cfg, small_graph):
        env = make_env(cfg, small_graph, delay=0, reward_correct=1.0, reward_incorrect=0.0)
        env.reset(4)
        targets = env.current_targets()
        wrong = (targets + 1) % env.num_actions
        _, rewards, _, _ = env.step(wrong)
        assert torch.all(rewards == 0.0)

    def test_determinism_with_seed(self, cfg, small_graph):
        e1 = make_env(cfg, small_graph, delay=0)
        e2 = make_env(cfg, small_graph, delay=0)
        e1.reset(4)
        e2.reset(4)
        assert torch.equal(e1.sample_cues(10), e2.sample_cues(10))


class TestDelay:
    def test_targets_masked_during_delay(self, cfg, small_graph):
        env = make_env(cfg, small_graph, delay=2)
        env.reset(4)
        t0 = env.current_targets()  # delay=2, within_trial=0 -> masked
        assert torch.all(t0 == IGNORE_INDEX)

    def test_target_active_at_response_step(self, cfg, small_graph):
        env = make_env(cfg, small_graph, delay=2)
        env.reset(4)
        cues = env._cues.clone()
        env.step(torch.zeros(4, dtype=torch.long))  # within_trial 1
        env.step(torch.zeros(4, dtype=torch.long))  # within_trial 2 == delay
        assert torch.equal(env.current_targets(), env.mapping_tensor()[cues])

    def test_reward_only_at_response_step(self, cfg, small_graph):
        env = make_env(cfg, small_graph, delay=2, reward_correct=1.0)
        env.reset(4)
        _, r0, _, _ = env.step(torch.zeros(4, dtype=torch.long))
        _, r1, _, _ = env.step(torch.zeros(4, dtype=torch.long))
        assert torch.all(r0 == 0.0)
        assert torch.all(r1 == 0.0)
        # landing action equal to target gives reward at response step
        targets = env.current_targets()
        _, r2, _, _ = env.step(targets)
        assert torch.all(r2 == 1.0)


class TestReversal:
    def test_reverse_flips_mapping(self, cfg, small_graph):
        env = make_env(cfg, small_graph, mapping="cyclic")
        before = env.mapping_tensor()
        env.reverse()
        after = env.mapping_tensor()
        assert torch.equal(after, (before + 1) % env.num_actions)
        env.reverse()
        assert torch.equal(env.mapping_tensor(), before)

    def test_reward_follows_new_mapping_after_reversal(self, cfg, small_graph):
        env = make_env(cfg, small_graph, delay=0, mapping="cyclic")
        env.reset(4)
        env.reverse()
        targets = env.current_targets()
        _, rewards, _, _ = env.step(targets.clone())
        assert torch.all(rewards == 1.0)


class TestStochastic:
    def test_stochastic_prob_one_deterministically_inverts(self, cfg, small_graph):
        env = make_env(
            cfg, small_graph, delay=0, reward_correct=1.0, reward_incorrect=0.0,
            stochastic_prob=1.0,
        )
        env.reset(4)
        targets = env.current_targets()
        _, rewards, _, _ = env.step(targets.clone())
        assert torch.all(rewards == 0.0)  # correct answers now unrewarded


class TestTasks:
    def test_hierarchical_features_share_family_structure(self):
        feats, fam = make_cue_features(6, 8, structure="hierarchical", num_families=2, seed=3)
        assert feats.shape == (6, 8)
        assert fam.tolist() == [0, 1, 0, 1, 0, 1]
        within = torch.cosine_similarity(feats[0], feats[2], dim=0)
        across = torch.cosine_similarity(feats[0], feats[1], dim=0)
        assert within > across

    def test_family_mapping(self):
        _, fam = make_cue_features(4, 4, structure="hierarchical", num_families=2, seed=0)
        m = make_mapping(4, 2, "family", fam)
        assert m.tolist() == [0, 1, 0, 1]
