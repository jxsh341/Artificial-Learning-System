"""Graph-dependent classification environment (immediate + delayed variants).

Trial structure (``delay + 1`` steps):

- step 0: a batch of graph instances is presented (markers visible);
- steps 1..delay-1: blank observations — the marker channel is cleared
  (making the answer uncomputable from the input) unless
  ``keep_markers_during_delay`` is set;
- step ``delay``: response required; target = instance label.

With ``delay = 0`` the presentation and response coincide (purely reactive
relational task). With ``delay > 0`` the system must compute the relational
answer from the graph and preserve it in temporal state — the two
requirements are independently ablatable.
"""
from __future__ import annotations

import torch
from torch import Tensor

from artificial_system.data.graph_tasks import ConnectivityDataset
from artificial_system.data.schema import GraphData
from artificial_system.environment.base import Environment

IGNORE_INDEX = -100


class GraphClassificationEnvironment(Environment):
    def __init__(
        self,
        dataset: ConnectivityDataset,
        config: dict,
        seed: int = 0,
    ):
        cfg = dict(config)
        self.dataset = dataset
        self.delay = int(cfg.get("delay", 0))
        self.reward_correct = float(cfg.get("reward_correct", 1.0))
        self.reward_incorrect = float(cfg.get("reward_incorrect", 0.0))
        self.stochastic_prob = float(cfg.get("stochastic_prob", 0.0))
        self.episode_length = cfg.get("episode_length")  # None => trainer-controlled
        self.marker_dim = int(cfg.get("marker_dim", 2))
        self.keep_markers_during_delay = bool(cfg.get("keep_markers_during_delay", False))
        self.blank_mode = str(cfg.get("blank_mode", "no_markers"))
        # "no_markers": blank steps show the current instance graph with the
        #   marker channel cleared (instance structure still visible);
        # "generic": blank steps show ONE fixed shared graph (no markers,
        #   zero features, fixed topology) — no instance information ever
        #   reaches the model outside the presentation step, so the only
        #   route from presentation to response is through temporal state.
        self.rng = torch.Generator().manual_seed(int(seed))
        self._batch_size = 0
        self._idxs = torch.zeros(1, dtype=torch.long)
        self._within_trial = 0
        self._episode_step = 0
        self._trial_count = 0
        self._generic_graph: GraphData | None = None
        if self.blank_mode == "generic":
            import networkx as nx

            n = self.dataset.graphs[0].num_nodes
            g = nx.gnp_random_graph(n, 0.05, seed=int(seed) + 99991)
            if not nx.is_connected(g):
                comps = [sorted(c) for c in nx.connected_components(g)]
                for a, b in zip(comps, comps[1:]):
                    g.add_edge(a[0], b[0])
            edges = list(g.edges())
            edges = edges + [(v, u) for (u, v) in edges]
            self._generic_graph = GraphData(
                node_features=torch.zeros(n, self.dataset.graphs[0].node_feature_dim),
                edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
                metadata={"blank": "generic"},
            ).validate()

    # ------------------------------------------------------------------ API

    @property
    def num_actions(self) -> int:
        return 2  # binary: connected / not connected

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def trial_count(self) -> int:
        return self._trial_count

    def sample_instances(self, batch_size: int) -> Tensor:
        return torch.randint(len(self.dataset), (batch_size,), generator=self.rng)

    def instance_observation(self, idxs) -> GraphData:
        """Observation for explicit instances, markers visible."""
        return self._collate([self.dataset.graphs[int(i)] for i in idxs])

    def blank_observation(self, keep_markers: bool | None = None, idxs=None) -> GraphData:
        """Observation for the current (or explicit) instances with the
        marker channel cleared — the answer is uncomputable from input."""
        idxs = self._idxs if idxs is None else idxs
        keep = self.keep_markers_during_delay if keep_markers is None else keep_markers
        if keep:
            return self.instance_observation(idxs)
        if self.blank_mode == "generic" and self._generic_graph is not None:
            return self._collate([self._generic_graph] * len(list(idxs)))
        graphs = []
        for i in idxs:
            g = self.dataset.graphs[int(i)]
            feats = g.node_features.clone()
            feats[:, : self.marker_dim] = 0.0
            graphs.append(GraphData(node_features=feats, edge_index=g.edge_index, metadata=dict(g.metadata)))
        return self._collate(graphs)

    def labels_for(self, idxs) -> Tensor:
        return self.dataset.labels[torch.as_tensor(list(idxs), dtype=torch.long)].clone()

    def reset(self, batch_size: int) -> GraphData:
        self._batch_size = int(batch_size)
        self._within_trial = 0
        self._episode_step = 0
        self._trial_count = 0
        self._idxs = self.sample_instances(self._batch_size)
        return self.instance_observation(self._idxs)

    def current_targets(self) -> Tensor:
        if self._within_trial != self.delay:
            return torch.full((self._batch_size,), IGNORE_INDEX, dtype=torch.long)
        return self.labels_for(self._idxs)

    def step(self, actions: Tensor) -> tuple[GraphData, Tensor, Tensor, dict]:
        actions = torch.as_tensor(actions, dtype=torch.long)
        if actions.shape[0] != self._batch_size:
            raise ValueError("actions must match batch size")
        active = self._within_trial == self.delay
        rewards = torch.zeros(self._batch_size)
        correct = torch.zeros(self._batch_size, dtype=torch.bool)
        if active:
            targets = self.labels_for(self._idxs)
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
            "instances": self._idxs.clone(),
            "correct": correct,
            "targets": self.labels_for(self._idxs),
        }

        self._episode_step += 1
        self._within_trial += 1
        if self._within_trial > self.delay:
            self._within_trial = 0
            self._idxs = self.sample_instances(self._batch_size)
            self._trial_count += 1

        done = torch.zeros(self._batch_size, dtype=torch.bool)
        if self.episode_length is not None and self._episode_step >= self.episode_length:
            done[:] = True

        return self._current_observation(), rewards, done, info

    # -------------------------------------------------------------- internals

    def _current_observation(self) -> GraphData:
        """Cue step 0: full instance with markers; delay/response steps: blank."""
        if self._within_trial == 0:
            return self.instance_observation(self._idxs)
        return self.blank_observation()

    def _collate(self, graphs: list[GraphData]) -> GraphData:
        from artificial_system.data.loaders import collate_graphs

        return collate_graphs(graphs)
