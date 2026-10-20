"""Message-passing GNN encoder (spec §7).

Pure-PyTorch implementation (no PyG dependency):

    h_i^(l+1) = UPDATE(h_i^l, AGGREGATE_j(MESSAGE(h_i^l, h_j^l, e_ij)))

with configurable aggregation (mean | sum | max), residual connections,
and layer normalization.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from artificial_system.data.schema import GraphData

AGGREGATIONS = ("mean", "sum", "max")


def scatter_aggregate(src: Tensor, index: Tensor, dim_size: int, reduce: str = "mean") -> Tensor:
    """Aggregate ``src[E, D]`` into ``out[dim_size, D]`` along ``index[E]``."""
    if src.shape[0] == 0:
        return torch.zeros(dim_size, src.shape[-1], device=src.device, dtype=src.dtype)
    if reduce in ("sum", "mean"):
        out = torch.zeros(dim_size, src.shape[-1], device=src.device, dtype=src.dtype)
        out.index_add_(0, index, src)
        if reduce == "mean":
            counts = torch.bincount(index, minlength=dim_size).clamp(min=1).to(src.dtype)
            out = out / counts.unsqueeze(-1)
        return out
    if reduce == "max":
        idx = index.unsqueeze(-1).expand(-1, src.shape[-1])
        out = torch.full(
            (dim_size, src.shape[-1]),
            torch.finfo(src.dtype).min,
            device=src.device,
            dtype=src.dtype,
        )
        out.scatter_reduce_(0, idx, src, reduce="amax", include_self=True)
        counts = torch.bincount(index, minlength=dim_size)
        empty = (counts == 0).unsqueeze(-1)
        return torch.where(empty, torch.zeros_like(out), out)
    raise ValueError(f"unknown aggregation {reduce!r}; choose from {AGGREGATIONS}")


class MessagePassingLayer(nn.Module):
    """One round of message passing: message -> aggregate -> update."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int = 0,
        aggregation: str = "mean",
    ):
        super().__init__()
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}")
        self.aggregation = aggregation
        self.lin_message = nn.Linear(in_dim, out_dim)
        self.lin_edge = nn.Linear(edge_dim, out_dim, bias=False) if edge_dim > 0 else None
        self.lin_update = nn.Linear(in_dim + out_dim, out_dim)

    def forward(
        self,
        h: Tensor,
        edge_index: Tensor,
        edge_features: Tensor | None = None,
    ) -> Tensor:
        n = h.shape[0]
        if edge_index.shape[1] == 0:
            agg = torch.zeros(n, self.lin_message.out_features, device=h.device, dtype=h.dtype)
        else:
            src, dst = edge_index[0], edge_index[1]
            msg = self.lin_message(h[src])
            if self.lin_edge is not None and edge_features is not None:
                msg = msg + self.lin_edge(edge_features)
            agg = scatter_aggregate(msg, dst, n, self.aggregation)
        return self.lin_update(torch.cat([h, agg], dim=-1))


class GraphEncoder(nn.Module):
    """Input projection -> L message-passing layers -> node representations."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        layers: int = 2,
        edge_dim: int = 0,
        aggregation: str = "mean",
        residual: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("GraphEncoder needs at least 1 layer")
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.uses_edge_features = edge_dim > 0
        self.residual = residual
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            MessagePassingLayer(hidden_dim, hidden_dim, edge_dim, aggregation)
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph: GraphData) -> Tensor:
        """GraphData -> node representations ``[N, hidden_dim]``."""
        h = self.input_proj(graph.node_features)
        edge_features = graph.edge_features if self.uses_edge_features else None
        mask_f = None
        if graph.node_mask is not None:
            mask_f = graph.node_mask.unsqueeze(-1).to(h.dtype)
            h = h * mask_f  # silenced nodes contribute no outgoing messages
        for layer, norm in zip(self.layers, self.norms):
            update = self.dropout(self.act(norm(layer(h, graph.edge_index, edge_features))))
            h = h + update if self.residual else update
            if mask_f is not None:
                h = h * mask_f  # keep silenced nodes silent through depth
        return h


class IdentityEncoder(nn.Module):
    """Pass-through encoder exposing raw node features (ablation: no graph computation)."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.hidden_dim = in_dim

    def forward(self, graph: GraphData) -> Tensor:
        h = graph.node_features
        if graph.node_mask is not None:
            h = h * graph.node_mask.unsqueeze(-1).to(h.dtype)
        return h


class MLPEncoder(nn.Module):
    """Per-node MLP with no message passing (ablation M6: tests whether structure matters)."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        modules: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(max(0, layers - 1)):
            modules += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        modules.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*modules)

    def forward(self, graph: GraphData) -> Tensor:
        h = self.net(graph.node_features)
        if graph.node_mask is not None:
            h = h * graph.node_mask.unsqueeze(-1).to(h.dtype)
        return h
