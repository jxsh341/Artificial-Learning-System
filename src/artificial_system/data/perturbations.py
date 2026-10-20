"""Graph perturbations (spec §19).

Every perturbation returns a new validated :class:`GraphData` through the
same stable interface — nothing downstream changes. Perturbations attack
relational structure and node identity; the cue channel remains
input-driven by design (the environment writes it after perturbation), so
these probes measure reliance on structure *beyond* the raw cue.

Uniform signature: ``fn(graph, level, seed) -> GraphData``. ``level`` is
ignored by perturbations that have no natural magnitude (shuffle,
drop_community).
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import torch

from artificial_system.data.schema import GraphData

GRAPH_PERTURBATIONS = (
    "remove_edges",
    "randomize_edges",
    "perturb_weights",
    "shuffle_features",
    "remove_nodes",
    "drop_community",
)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _rebuild(graph: GraphData, **overrides) -> GraphData:
    fields = dict(
        node_features=graph.node_features,
        edge_index=graph.edge_index,
        edge_features=graph.edge_features,
        node_mask=graph.node_mask,
        batch=graph.batch,
        num_graphs=graph.num_graphs,
        metadata={**graph.metadata, **overrides.pop("metadata_extra", {})},
    )
    fields.update(overrides)
    return GraphData(**fields).validate()


def remove_edges(graph: GraphData, fraction: float, seed: int = 0) -> GraphData:
    """Randomly delete a fraction of edges (always keeps at least one)."""
    e = graph.num_edges
    if e == 0 or fraction <= 0:
        return _rebuild(graph, metadata_extra={"perturbation": "remove_edges", "level": fraction})
    keep = _rng(seed).random(e) >= float(fraction)
    if not keep.any():
        keep[int(_rng(seed).integers(e))] = True
    keep_t = torch.from_numpy(keep.astype(bool))
    return _rebuild(
        graph,
        edge_index=graph.edge_index[:, keep_t],
        edge_features=None if graph.edge_features is None else graph.edge_features[keep_t],
        metadata_extra={"perturbation": "remove_edges", "level": fraction, "seed": seed},
    )


def randomize_edges(graph: GraphData, fraction: float, seed: int = 0) -> GraphData:
    """Rewire a fraction of edge targets to random destinations."""
    e, n = graph.num_edges, graph.num_nodes
    ei = graph.edge_index.clone()
    if e > 0 and fraction > 0:
        rng = _rng(seed)
        k = int(round(float(fraction) * e))
        idx = rng.choice(e, size=min(k, e), replace=False)
        ei[1, torch.from_numpy(idx.astype(np.int64))] = torch.from_numpy(
            rng.integers(0, n, size=len(idx)).astype(np.int64)
        )
    return _rebuild(graph, edge_index=ei,
                    metadata_extra={"perturbation": "randomize_edges", "level": fraction, "seed": seed})


def swap_edges(graph: GraphData, fraction: float, seed: int = 0) -> GraphData:
    """Degree-preserving double-edge swaps (``level`` = fraction of edges).

    Unlike :func:`randomize_edges`, the degree sequence is preserved exactly —
    so any behavioural change under this perturbation cannot be attributed to
    degree statistics or local density, only to relational structure.
    """
    import networkx as nx

    e = graph.num_edges
    if e < 4 or fraction <= 0:
        return _rebuild(graph, metadata_extra={"perturbation": "swap_edges", "level": fraction})
    und = nx.Graph()
    und.add_nodes_from(range(graph.num_nodes))
    und.add_edges_from({(int(u), int(v)) for u, v in graph.edge_index.t().tolist()})
    rng = np.random.default_rng(seed)
    try:
        nx.double_edge_swap(und, nswap=max(1, int(round(float(fraction) * und.number_of_edges() / 2))),
                            max_tries=100 * und.number_of_edges(), seed=int(rng.integers(0, 2**31 - 1)))
    except nx.NetworkXError:
        pass  # keep original topology if swaps are infeasible
    edges = list(und.edges())
    edges = edges + [(v, u) for (u, v) in edges]  # keep both directions
    return _rebuild(
        graph,
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        metadata_extra={"perturbation": "swap_edges", "level": fraction, "seed": seed},
    )


def perturb_weights(graph: GraphData, level: float, seed: int = 0) -> GraphData:
    """Multiplicatively perturb edge weights/features: ``w *= N(1, level)``."""
    if graph.edge_features is None or level <= 0:
        return _rebuild(graph, metadata_extra={"perturbation": "perturb_weights", "level": level})
    rng = _rng(seed)
    scale = rng.normal(1.0, float(level), graph.edge_features.shape).astype(np.float32)
    scale = np.clip(scale, 0.05, None)
    return _rebuild(
        graph,
        edge_features=graph.edge_features * torch.from_numpy(scale),
        metadata_extra={"perturbation": "perturb_weights", "level": level, "seed": seed},
    )


def shuffle_features(graph: GraphData, level: float = 1.0, seed: int = 0) -> GraphData:
    """Permute node-feature rows: destroys node identity, keeps graph topology."""
    perm = _rng(seed).permutation(graph.num_nodes)
    return _rebuild(
        graph,
        node_features=graph.node_features[torch.from_numpy(perm.astype(np.int64))],
        metadata_extra={"perturbation": "shuffle_features", "seed": seed},
    )


def remove_nodes(graph: GraphData, fraction: float, seed: int = 0) -> GraphData:
    """Silence a fraction of nodes (``node_mask = False``): features still
    present in the tensor but excluded from encoder computation."""
    n = graph.num_nodes
    mask = torch.ones(n, dtype=torch.bool)
    if fraction > 0 and n > 0:
        rng = _rng(seed)
        k = min(n, int(round(float(fraction) * n)))
        idx = rng.choice(n, size=k, replace=False)
        mask[torch.from_numpy(idx.astype(np.int64))] = False
    return _rebuild(graph, node_mask=mask,
                    metadata_extra={"perturbation": "remove_nodes", "level": fraction, "seed": seed})


def drop_community(graph: GraphData, level: float = 1.0, seed: int = 0) -> GraphData:
    """Silence one detected community (population) of nodes."""
    n = graph.num_nodes
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(map(tuple, graph.edge_index.t().tolist()))
    mask = torch.ones(n, dtype=torch.bool)
    label = "highest_degree_fallback"
    communities = nx.community.greedy_modularity_communities(g)
    if communities:
        comm = sorted(communities[int(_rng(seed).integers(len(communities)))])
        mask[torch.tensor(comm, dtype=torch.long)] = False
        label = f"community_size_{len(comm)}"
    else:  # fallback: silence the highest-degree nodes
        deg = torch.bincount(graph.edge_index[0], minlength=n) + torch.bincount(
            graph.edge_index[1], minlength=n
        )
        k = max(1, n // 10)
        mask[deg.topk(k).indices] = False
    return _rebuild(graph, node_mask=mask,
                    metadata_extra={"perturbation": "drop_community", "target": label, "seed": seed})


def perturb_graph(graph: GraphData, kind: str, level: float, seed: int = 0) -> GraphData:
    """Dispatch by name (used by experiment configs)."""
    table = {name: fn for name, fn in globals().items() if name in GRAPH_PERTURBATIONS}
    if kind not in table:
        raise ValueError(f"unknown graph perturbation {kind!r}; choose from {GRAPH_PERTURBATIONS}")
    return table[kind](graph, level, seed)
