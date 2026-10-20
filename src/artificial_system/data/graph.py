"""Synthetic graph generators (spec §6).

All families are produced through exactly the same :class:`GraphData`
interface: ``random``, ``regular``, ``small_world``, ``scale_free``,
``modular``.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import torch

from artificial_system.data.schema import GraphData

FAMILY_NAMES = ("random", "regular", "small_world", "scale_free", "modular")


def _build_topology(
    family: str, num_nodes: int, rng: np.random.Generator, params: dict
) -> nx.Graph:
    seed_int = int(rng.integers(0, 2**31 - 1))
    n = num_nodes
    if family == "random":
        p = float(params.get("p", min(1.0, 4.0 / max(n - 1, 1))))
        g = nx.gnp_random_graph(n, p, seed=seed_int)
    elif family == "regular":
        k = int(params.get("k", 4))
        k = max(2, min(k, n - 1))
        if (n * k) % 2 == 1:
            k -= 1
        g = nx.random_regular_graph(k, n, seed=seed_int)
    elif family == "small_world":
        k = int(params.get("k", 6))
        k = max(2, min(k, n - 1))
        if k % 2 == 1:
            k = k - 1 if k > 2 else k + 1
        k = min(k, n - 1 - (n - 1) % 2) if (n - 1) % 2 == 0 else min(k, n - 2)
        k = max(2, k - (k % 2))
        p = float(params.get("p", 0.1))
        g = nx.watts_strogatz_graph(n, k, p, seed=seed_int)
    elif family == "scale_free":
        m = int(params.get("m", 2))
        m = max(1, min(m, n - 1))
        g = nx.barabasi_albert_graph(n, m, seed=seed_int)
    elif family == "modular":
        k = int(params.get("communities", 4))
        k = max(1, min(k, n))
        base, rem = divmod(n, k)
        sizes = [base + 1] * rem + [base] * (k - rem)
        p_in = float(params.get("p_in", 0.2))
        p_out = float(params.get("p_out", 0.02))
        g = nx.random_partition_graph(sizes, p_in, p_out, seed=seed_int)
    else:
        raise ValueError(f"unknown graph family {family!r}; choose from {FAMILY_NAMES}")

    # Guarantee weak connectivity so information can always propagate.
    if g.number_of_nodes() > 1 and not nx.is_connected(g):
        comps = [sorted(c) for c in nx.connected_components(g)]
        for a, b in zip(comps, comps[1:]):
            g.add_edge(a[0], b[0])
    return g


def generate_graph(
    family: str,
    num_nodes: int,
    node_feature_dim: int,
    edge_feature_dim: int = 0,
    directed: bool = True,
    weighted: bool = False,
    seed: int = 0,
    params: dict | None = None,
) -> GraphData:
    """Generate one synthetic graph of the requested family."""
    params = dict(params or {})
    rng = np.random.default_rng(seed)
    g = _build_topology(family, num_nodes, rng, params)

    undirected_edges = list(g.edges())
    if directed:
        edges = [
            (u, v) if rng.random() < 0.5 else (v, u) for (u, v) in undirected_edges
        ]
    else:
        edges = [(u, v) for (u, v) in undirected_edges] + [
            (v, u) for (u, v) in undirected_edges
        ]

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:  # degenerate single-node graph: keep a self loop so E > 0 downstream ops stay simple
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    num_edges = edge_index.shape[1]
    node_features = torch.from_numpy(
        rng.standard_normal((num_nodes, node_feature_dim)).astype(np.float32)
    )

    edge_features = None
    if edge_feature_dim > 0:
        feats = rng.standard_normal((num_edges, edge_feature_dim)).astype(np.float32)
        if weighted:
            feats[:, 0] = rng.uniform(0.5, 1.5, num_edges).astype(np.float32)
        edge_features = torch.from_numpy(feats)
    elif weighted:
        edge_features = torch.from_numpy(
            rng.uniform(0.5, 1.5, (num_edges, 1)).astype(np.float32)
        )

    return GraphData(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        metadata={
            "family": family,
            "num_nodes": num_nodes,
            "directed": directed,
            "weighted": weighted,
            "seed": seed,
            "params": params,
        },
    ).validate()


class SyntheticGraphGenerator:
    """Config-driven generator producing graphs through a stable interface."""

    def __init__(self, config: dict, family: str | None = None):
        self.config = dict(config)
        self.family = family or self.config.get("family", "random")

    def generate(self, family: str | None = None, seed: int | None = None) -> GraphData:
        cfg = self.config
        return generate_graph(
            family=family or self.family,
            num_nodes=int(cfg.get("num_nodes", 100)),
            node_feature_dim=int(cfg.get("node_feature_dim", 16)),
            edge_feature_dim=int(cfg.get("edge_feature_dim", 0)),
            directed=bool(cfg.get("directed", True)),
            weighted=bool(cfg.get("weighted", False)),
            seed=int(cfg.get("seed", 0) if seed is None else seed),
            params=cfg.get("params"),
        )
