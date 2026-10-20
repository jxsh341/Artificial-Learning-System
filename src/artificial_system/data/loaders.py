"""Batching utilities for GraphData (disjoint-union batching, spec §5).

Collating B graphs produces a single :class:`GraphData` with:

- concatenated node features ``[sum(N_i), F]``,
- offset ``edge_index``,
- a ``batch`` vector ``[sum(N_i)]`` assigning each node to a graph,
- ``num_graphs = B``.
"""
from __future__ import annotations

import torch

from artificial_system.data.schema import GraphData


def collate_graphs(graphs: list[GraphData]) -> GraphData:
    """Collate a list of GraphData into one batched GraphData."""
    if not graphs:
        raise ValueError("cannot collate an empty list of graphs")
    feature_dim = graphs[0].node_feature_dim
    has_edge_feat = graphs[0].edge_features is not None
    has_mask = graphs[0].node_mask is not None
    for g in graphs:
        g.validate()
        if g.batch is not None:
            raise ValueError("collate expects single (unbatched) graphs")
        if g.node_feature_dim != feature_dim:
            raise ValueError("all graphs must share node feature dimension")
        if (g.edge_features is not None) != has_edge_feat:
            raise ValueError("edge features must be present on all or none of the graphs")
        if (g.node_mask is not None) != has_mask:
            raise ValueError("node_mask must be present on all or none of the graphs")

    node_features = torch.cat([g.node_features for g in graphs], dim=0)
    edge_indices, edge_features, node_masks, batch = [], [], [], []
    offset = 0
    for i, g in enumerate(graphs):
        edge_indices.append(g.edge_index + offset)
        if has_edge_feat:
            edge_features.append(g.edge_features)
        if has_mask:
            node_masks.append(g.node_mask)
        batch.append(torch.full((g.num_nodes,), i, dtype=torch.long))
        offset += g.num_nodes

    return GraphData(
        node_features=node_features,
        edge_index=torch.cat(edge_indices, dim=1),
        edge_features=torch.cat(edge_features, dim=0) if has_edge_feat else None,
        node_mask=torch.cat(node_masks, dim=0) if has_mask else None,
        batch=torch.cat(batch, dim=0),
        num_graphs=len(graphs),
        metadata={"graphs": [g.metadata for g in graphs]},
    ).validate()


def unbatch_graph(graph: GraphData) -> list[GraphData]:
    """Split a batched GraphData back into per-graph GraphData objects."""
    graph.validate()
    if graph.batch is None or graph.num_graphs == 1:
        return [graph]
    out: list[GraphData] = []
    for i in range(graph.num_graphs):
        node_sel = (graph.batch == i).nonzero(as_tuple=True)[0]
        node_set_min = int(node_sel.min())
        node_set = set(node_sel.tolist())
        src, dst = graph.edge_index[0], graph.edge_index[1]
        keep = torch.tensor(
            [int(s) in node_set and int(d) in node_set for s, d in zip(src.tolist(), dst.tolist())],
            dtype=torch.bool,
        )
        ei = graph.edge_index[:, keep] - node_set_min
        out.append(
            GraphData(
                node_features=graph.node_features[node_sel],
                edge_index=ei,
                edge_features=graph.edge_features[keep] if graph.edge_features is not None else None,
                node_mask=graph.node_mask[node_sel] if graph.node_mask is not None else None,
                metadata=dict(graph.metadata.get("graphs", [{}])[i])
                if "graphs" in graph.metadata
                else {},
            )
        )
    return out
