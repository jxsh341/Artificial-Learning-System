"""Graph-dependent classification tasks: make topology causally necessary.

Constructs graph instances in which the label depends on connectivity
between two marked nodes — not on the mere presence of a cue. Controls
against non-relational shortcuts:

- node features are iid per instance from one distribution across classes;
  the two markers (A, B) are the only class-asymmetric feature and their
  placement is random, so memorising node identities cannot work;
- node counts are identical and edge counts are approximately matched
  across classes (dataset statistics expose residual confounds);
- instances are undirected (both edge directions in ``edge_index``) and
  labels are computed on the undirected projection, so a message-passing
  GNN can in principle traverse every true path;
- positives are restricted to marked pairs at distance <= ``max_path_len``
  (detectable with a GNN of depth ceil(max_path_len / 2)); negatives place
  the marked nodes in different components.

The decisive external control is rewiring test edges with/without label
recomputation — see ``evaluation/graph_classification.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import torch
from torch import Tensor

from artificial_system.data.schema import GraphData


@dataclass
class ConnectivityDataset:
    """A dataset of graph instances with binary connectivity labels."""

    graphs: list[GraphData]
    labels: Tensor  # [num_instances] long in {0, 1}
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.graphs)


def _instance_features(
    n: int, feature_dim: int, marker_dim: int, marked: list[int], rng: np.random.Generator
) -> Tensor:
    feats = rng.standard_normal((n, feature_dim)).astype(np.float32)
    feats[:, :marker_dim] = 0.0  # marker channel clean on all non-marked nodes
    for slot, node in enumerate(marked):  # marker A -> slot 0, marker B -> slot 1
        feats[node, slot % marker_dim] = 1.0
    return torch.from_numpy(feats)


def _connect_components(g: nx.Graph) -> None:
    """Add edges so the graph becomes connected (in place)."""
    if g.number_of_nodes() > 1 and not nx.is_connected(g):
        comps = [sorted(c) for c in nx.connected_components(g)]
        for a, b in zip(comps, comps[1:]):
            g.add_edge(a[0], b[0])


def _make_instance(
    label: int,
    num_nodes: int,
    label_mode: str,
    max_path_len: int,
    neg_min_distance: int,
    edge_density: float,
    feature_dim: int,
    marker_dim: int,
    directed: bool,
    rng: np.random.Generator,
) -> tuple[GraphData, int, dict]:
    """Build one instance. BOTH classes come from the same connected-graph
    family — only the marked pair differs — so no graph-level statistic
    (component count, degree distribution, density) can reveal the label:

    - ``adjacency``: positives = marked pair adjacent (distance 1);
      negatives = marked pair at distance >= ``neg_min_distance``.
    - ``path``: positives = marked pair at distance <= ``max_path_len``;
      negatives = marked pair at distance > ``max_path_len``.
    """
    if num_nodes < 4:
        raise ValueError(f"num_nodes={num_nodes} too small")
    attempts = 0
    while True:
        attempts += 1
        seed_int = int(rng.integers(0, 2**31 - 1))
        g = nx.gnp_random_graph(num_nodes, edge_density, seed=seed_int)
        if not nx.is_connected(g):
            continue
        sp = dict(nx.all_pairs_shortest_path_length(g))
        all_pairs = [(u, v) for u in range(num_nodes) for v in range(u + 1, num_nodes)]

        if label_mode == "adjacency":
            if label == 1:
                edges = list(g.edges())
                if not edges:
                    continue
                marked = list(edges[int(rng.integers(len(edges)))])
            else:
                candidates = [
                    (u, v) for (u, v) in all_pairs if sp[u].get(v, 10**9) >= neg_min_distance
                ]
                if not candidates:
                    continue
                marked = list(candidates[int(rng.integers(len(candidates)))])
        elif label_mode == "path":
            if label == 1:
                candidates = [
                    (u, v) for (u, v) in all_pairs if sp[u].get(v, 10**9) <= max_path_len
                ]
            else:
                candidates = [
                    (u, v) for (u, v) in all_pairs if sp[u].get(v, 10**9) > max_path_len
                ]
            if not candidates:
                continue
            marked = list(candidates[int(rng.integers(len(candidates)))])
        else:
            raise ValueError(f"unknown label_mode {label_mode!r}")
        path_len = int(sp[marked[0]].get(marked[1]))
        break

    edges = list(g.edges())
    if directed:
        edges = [
            (u, v) if rng.random() < 0.5 else (v, u) for (u, v) in edges
        ]
    else:
        edges = edges + [(v, u) for (u, v) in edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    marked_degree = float(np.mean([g.degree(u) for u in marked]))
    graph = GraphData(
        node_features=_instance_features(num_nodes, feature_dim, marker_dim, marked, rng),
        edge_index=edge_index,
        metadata={
            "label": label,
            "marked": marked,
            "path_len": path_len,
            "num_nodes": num_nodes,
            "num_edges": len(edges),
        },
    ).validate()
    info = {
        "label": label,
        "path_len": path_len,
        "num_nodes": num_nodes,
        "num_edges": len(edges),
        "marked_degree_mean": marked_degree,
        "attempts": attempts,
    }
    return graph, label, info


def make_connectivity_dataset(
    num_instances: int,
    num_nodes: int = 100,
    feature_dim: int = 16,
    marker_dim: int = 2,
    max_path_len: int = 3,
    edge_density: float = 0.05,
    directed: bool = False,
    seed: int = 0,
    positive_fraction: float = 0.5,
    label_mode: str = "path",
    neg_min_distance: int = 3,
) -> ConnectivityDataset:
    """Generate a balanced connectivity-classification dataset.

    ``label_mode`` (see :func:`_make_instance`): both classes come from the
    same connected-graph family; only the marked pair differs, so no
    graph-level statistic can reveal the label.
    """
    if marker_dim < 2:
        raise ValueError("marker_dim must be >= 2 (distinct markers A and B)")
    rng = np.random.default_rng(seed)
    graphs: list[GraphData] = []
    labels: list[int] = []
    infos: list[dict] = []
    for _ in range(num_instances):
        label = 1 if rng.random() < positive_fraction else 0
        g, lab, info = _make_instance(
            label, num_nodes, label_mode, max_path_len, neg_min_distance,
            edge_density, feature_dim, marker_dim, directed, rng,
        )
        graphs.append(g)
        labels.append(lab)
        infos.append(info)

    labels_t = torch.tensor(labels, dtype=torch.long)
    pos = [i for i in infos if i["label"] == 1]
    neg = [i for i in infos if i["label"] == 0]
    stats = {
        "num_instances": num_instances,
        "positive_fraction": len(pos) / max(num_instances, 1),
        "mean_edges_positive": float(np.mean([i["num_edges"] for i in pos])) if pos else float("nan"),
        "mean_edges_negative": float(np.mean([i["num_edges"] for i in neg])) if neg else float("nan"),
        "mean_marked_degree_positive": float(np.mean([i["marked_degree_mean"] for i in pos])) if pos else float("nan"),
        "mean_marked_degree_negative": float(np.mean([i["marked_degree_mean"] for i in neg])) if neg else float("nan"),
        "path_len_histogram": {
            str(d): sum(1 for i in infos if i["path_len"] == d)
            for d in sorted({i["path_len"] for i in infos if i["path_len"] is not None})
        },
    }
    return ConnectivityDataset(
        graphs=graphs,
        labels=labels_t,
        metadata={
            "config": {
                "num_nodes": num_nodes,
                "feature_dim": feature_dim,
                "marker_dim": marker_dim,
                "max_path_len": max_path_len,
                "edge_density": edge_density,
                "directed": directed,
                "seed": seed,
                "label_mode": label_mode,
                "neg_min_distance": neg_min_distance,
            },
            "instances": infos,
            "stats": stats,
        },
    )


def recompute_label(graph: GraphData, marked: list[int], max_path_len: int, label_mode: str = "path") -> int:
    """Label of a (possibly perturbed) graph on the undirected projection.

    - ``path``: 1 iff the marked nodes are connected by a path of length
      <= ``max_path_len``;
    - ``adjacency``: 1 iff the marked nodes are adjacent.
    """
    g = nx.Graph()
    g.add_nodes_from(range(graph.num_nodes))
    g.add_edges_from({(int(u), int(v)) for u, v in graph.edge_index.t().tolist()})
    u, v = int(marked[0]), int(marked[1])
    if u not in g or v not in g or not nx.has_path(g, u, v):
        return 0
    d = nx.shortest_path_length(g, u, v)
    if label_mode == "adjacency":
        return 1 if d == 1 else 0
    return 1 if d <= max_path_len else 0
