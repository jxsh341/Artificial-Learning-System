"""Configurable cue-learning environment (spec §12, §16).

Each trial proceeds as ``delay + 1`` steps:

- step 0: the cue is written into the first ``cue_feature_dim`` node-feature
  channels of the first ``sensory_nodes`` nodes of the base graph;
- steps 1..delay-1: blank observations;
- step ``delay``: blank observation, response required (target active,
  reward delivered).

With ``delay = 0`` the cue presentation and the response coincide, giving a
purely reactive task; larger delays require temporal memory. The cue ->
action mapping can be reversed on demand (spec §16).
"""
from __future__ import annotations

import torch
from torch import Tensor

from artificial_system.data.schema import GraphData
from artificial_system.environment.base import Environment
from artificial_system.environment.tasks import (
    make_cue_features,
    make_mapping,
    reversed_mapping,
)

IGNORE_INDEX = -100


class CueLearningEnvironment(Environment):
    def __init__(
        self,
        base_graph: GraphData,
        config: dict,
        cue_features: Tensor | None = None,
        family_ids: Tensor | None = None,
        allowed_cues: list[int] | None = None,
        seed: int = 0,
    ):
        cfg = dict(config)
        base_graph.validate()
        self.base_graph = base_graph
        self.num_cues = int(cfg.get("num_cues", 4))
        self._num_actions = int(cfg.get("num_actions", 2))
        self.cue_dim = int(cfg.get("cue_feature_dim", 8))
        self.delay = int(cfg.get("delay", 0))
        self.reward_correct = float(cfg.get("reward_correct", 1.0))
        self.reward_incorrect = float(cfg.get("reward_incorrect", 0.0))
        self.stochastic_prob = float(cfg.get("stochastic_prob", 0.0))
        self.episode_length = cfg.get("episode_length")  # None => controlled by the trainer
        n_sensory = cfg.get("sensory_nodes")
        self.sensory_nodes = (
            base_graph.num_nodes if n_sensory is None else int(n_sensory)
        )
        if self.cue_dim > base_graph.node_feature_dim:
            raise ValueError(
                f"cue_feature_dim ({self.cue_dim}) exceeds node_feature_dim "
                f"({base_graph.node_feature_dim})"
            )
        if not 1 <= self.sensory_nodes <= base_graph.num_nodes:
            raise ValueError("sensory_nodes out of range")

        if cue_features is None:
            cue_features, family_ids = make_cue_features(
                self.num_cues,
                self.cue_dim,
                structure=cfg.get("structure", "onehot"),
                num_families=int(cfg.get("num_families", 2)),
                feature_noise=float(cfg.get("feature_noise", 0.25)),
                seed=int(cfg.get("feature_seed", 12345)),
            )
        self.cue_features = cue_features
        self.family_ids = family_ids

        self._mapping = make_mapping(
            self.num_cues, self._num_actions, cfg.get("mapping", "cyclic"), family_ids
        )
        reversal = cfg.get("reversal_mapping")
        self._reversed_mapping = (
            make_mapping(self.num_cues, self._num_actions, reversal, family_ids)
            if reversal is not None
            else reversed_mapping(self._mapping, self._num_actions)
        )

        self._allowed = torch.as_tensor(
            list(allowed_cues) if allowed_cues is not None else list(range(self.num_cues)),
            dtype=torch.long,
        )
        if self._allowed.numel() == 0:
            raise ValueError("allowed_cues must not be empty")

        self.rng = torch.Generator().manual_seed(int(seed))
        self._batch_size = 0
        self._template_ready = False
        self._reset_episode_state()

    # ------------------------------------------------------------------ API

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def trial_count(self) -> int:
        return self._trial_count

    def mapping_tensor(self) -> Tensor:
        return self._mapping.clone()

    def set_mapping(self, mapping: Tensor) -> None:
        mapping = torch.as_tensor(mapping, dtype=torch.long)
        if mapping.shape[0] != self.num_cues:
            raise ValueError("mapping has wrong length")
        self._mapping = mapping

    def reverse(self) -> None:
        """Switch to the reversed cue -> action mapping (spec §16)."""
        self._mapping, self._reversed_mapping = self._reversed_mapping, self._mapping

    def sample_cues(self, batch_size: int) -> Tensor:
        idx = torch.randint(self._allowed.shape[0], (batch_size,), generator=self.rng)
        return self._allowed[idx]

    def reset(self, batch_size: int) -> GraphData:
        self._batch_size = int(batch_size)
        self._build_template()
        self._reset_episode_state()
        self._cues = self.sample_cues(self._batch_size)
        return self._current_observation()

    def current_targets(self) -> Tensor:
        if self._within_trial != self.delay:
            return torch.full((self._batch_size,), IGNORE_INDEX, dtype=torch.long)
        return self._mapping[self._cues]

    def step(self, actions: Tensor) -> tuple[GraphData, Tensor, Tensor, dict]:
        actions = torch.as_tensor(actions, dtype=torch.long)
        if actions.shape[0] != self._batch_size:
            raise ValueError("actions must match batch size")
        active = self._within_trial == self.delay
        rewards = torch.zeros(self._batch_size)
        correct = torch.zeros(self._batch_size, dtype=torch.bool)
        if active:
            targets = self._mapping[self._cues]
            correct = actions == targets
            r = torch.where(
                correct,
                torch.full_like(rewards, self.reward_correct),
                torch.full_like(rewards, self.reward_incorrect),
            )
            if self.stochastic_prob > 0.0:
                flip = torch.rand(self._batch_size, generator=self.rng) < self.stochastic_prob
                r = torch.where(
                    flip,
                    torch.full_like(r, self.reward_correct + self.reward_incorrect) - r,
                    r,
                )
            rewards = r

        info = {
            "active": active,
            "cues": self._cues.clone(),
            "correct": correct,
            "targets": self._mapping[self._cues].clone(),
        }

        self._episode_step += 1
        self._within_trial += 1
        if self._within_trial > self.delay:
            self._within_trial = 0
            self._cues = self.sample_cues(self._batch_size)
            self._trial_count += 1

        done = torch.zeros(self._batch_size, dtype=torch.bool)
        if self.episode_length is not None and self._episode_step >= self.episode_length:
            done[:] = True

        return self._current_observation(), rewards, done, info

    # -------------------------------------------------------- observations

    def cue_observation(self, cue_ids: Tensor) -> GraphData:
        """Observation for an explicit batch of cues (used by evaluations)."""
        self._require_template()
        cue_ids = torch.as_tensor(cue_ids, dtype=torch.long)
        feats = self._feat_template.clone()
        feats[:, : self.sensory_nodes, : self.cue_dim] = self.cue_features[cue_ids].unsqueeze(1)
        return self._wrap_graph(feats, cue_ids)

    def blank_observation(self) -> GraphData:
        """Observation with the cue channel cleared (delays, retention waits)."""
        self._require_template()
        return self._wrap_graph(self._feat_template.clone(), None)

    # --------------------------------------------------------------- internals

    def _reset_episode_state(self) -> None:
        self._within_trial = 0
        self._episode_step = 0
        self._trial_count = 0
        self._cues = torch.zeros(max(self._batch_size, 1), dtype=torch.long)

    def _require_template(self) -> None:
        if not self._template_ready:
            raise RuntimeError("call reset(batch_size) before requesting observations")

    def _build_template(self) -> None:
        b, n = self._batch_size, self.base_graph.num_nodes
        feats = self.base_graph.node_features.unsqueeze(0).repeat(b, 1, 1)
        feats[:, : self.sensory_nodes, : self.cue_dim] = 0.0
        self._feat_template = feats
        offsets = torch.arange(b).view(b, 1, 1) * n
        ei = self.base_graph.edge_index.unsqueeze(0) + offsets  # [B, 2, E]
        self._edge_index = ei.permute(1, 0, 2).reshape(2, -1)
        ef = self.base_graph.edge_features
        self._edge_features = ef.repeat(b, 1) if ef is not None else None
        self._batch_vec = torch.repeat_interleave(torch.arange(b), n)
        self._template_ready = True

    def _wrap_graph(self, feats: Tensor, cue_ids: Tensor | None) -> GraphData:
        b, n, f = feats.shape
        return GraphData(
            node_features=feats.reshape(b * n, f),
            edge_index=self._edge_index,
            edge_features=self._edge_features,
            batch=self._batch_vec,
            num_graphs=b,
            metadata={"cues": cue_ids.tolist() if cue_ids is not None else None},
        )

    def _current_observation(self) -> GraphData:
        if self._within_trial == 0:
            return self.cue_observation(self._cues)
        return self.blank_observation()
