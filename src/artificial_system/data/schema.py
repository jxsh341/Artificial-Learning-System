"""Stable graph data interface (spec §5).

Every component in the system communicates with graph data exclusively
through :class:`GraphData`. No assumptions about what a node or an edge
represents are made anywhere outside of adapters/generators.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch
from torch import Tensor


@dataclass
class GraphData:
    """Minimal graph object.

    Attributes:
        node_features: Float tensor ``[N, F_node]``.
        edge_index: Long tensor ``[2, E]`` of directed edges (source, target).
        edge_features: Optional float tensor ``[E, F_edge]``.
        node_mask: Optional bool tensor ``[N]`` marking valid nodes.
        batch: Optional long tensor ``[N]`` assigning each node to a graph in
            a batched (disjoint-union) representation.
        num_graphs: Number of graphs represented (1 for a single graph).
        metadata: Free-form dictionary (kept out of the tensor interface).
    """

    node_features: Tensor
    edge_index: Tensor
    edge_features: Tensor | None = None
    node_mask: Tensor | None = None
    batch: Tensor | None = None
    num_graphs: int = 1
    metadata: dict = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def node_feature_dim(self) -> int:
        return int(self.node_features.shape[1])

    @property
    def edge_feature_dim(self) -> int:
        return 0 if self.edge_features is None else int(self.edge_features.shape[1])

    @property
    def device(self) -> torch.device:
        return self.node_features.device

    def validate(self) -> "GraphData":
        """Validate dimensions, dtypes, and indexing. Raises ``ValueError``."""
        if self.node_features.ndim != 2:
            raise ValueError(f"node_features must be [N, F], got {tuple(self.node_features.shape)}")
        if self.node_features.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"node_features must be float, got {self.node_features.dtype}")
        n = self.num_nodes
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must be [2, E], got {tuple(self.edge_index.shape)}")
        if self.edge_index.dtype != torch.long:
            raise ValueError(f"edge_index must be long, got {self.edge_index.dtype}")
        if self.num_edges > 0:
            lo = int(self.edge_index.min())
            hi = int(self.edge_index.max())
            if lo < 0 or hi >= n:
                raise ValueError(f"edge_index values out of range [0, {n}): [{lo}, {hi}]")
        if self.edge_features is not None:
            if self.edge_features.ndim != 2 or self.edge_features.shape[0] != self.num_edges:
                raise ValueError(
                    f"edge_features must be [E, F], got {tuple(self.edge_features.shape)} "
                    f"for E={self.num_edges}"
                )
        if self.node_mask is not None:
            if self.node_mask.shape != (n,):
                raise ValueError(f"node_mask must be [N], got {tuple(self.node_mask.shape)}")
        if self.batch is not None:
            if self.batch.shape != (n,):
                raise ValueError(f"batch must be [N], got {tuple(self.batch.shape)}")
            if self.num_edges >= 0 and self.batch.numel() > 0:
                if int(self.batch.min()) < 0 or int(self.batch.max()) >= self.num_graphs:
                    raise ValueError("batch indices out of range for num_graphs")
        if self.num_graphs < 1:
            raise ValueError("num_graphs must be >= 1")
        return self

    def to(self, device: torch.device | str) -> "GraphData":
        """Return a copy with all tensors moved to ``device``."""
        return replace(
            self,
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_features=None if self.edge_features is None else self.edge_features.to(device),
            node_mask=None if self.node_mask is None else self.node_mask.to(device),
            batch=None if self.batch is None else self.batch.to(device),
        )
