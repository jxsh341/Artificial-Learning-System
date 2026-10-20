"""Graph-level pooling and prediction readouts (spec §8).

Every pooling module maps node embeddings ``[N, H]`` + ``batch`` vector to a
per-graph representation ``[num_graphs, out_dim]`` and exposes ``out_dim``.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class MeanPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.out_dim = dim

    def forward(self, h: Tensor, batch: Tensor, size: int) -> Tensor:
        out = torch.zeros(size, h.shape[1], device=h.device, dtype=h.dtype)
        out.index_add_(0, batch, h)
        counts = torch.bincount(batch, minlength=size).clamp(min=1).unsqueeze(-1)
        return out / counts.to(h.dtype)


class MaxPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.out_dim = dim

    def forward(self, h: Tensor, batch: Tensor, size: int) -> Tensor:
        out = torch.full(
            (size, h.shape[1]), torch.finfo(h.dtype).min, device=h.device, dtype=h.dtype
        )
        idx = batch.unsqueeze(-1).expand(-1, h.shape[1])
        out.scatter_reduce_(0, idx, h, reduce="amax", include_self=True)
        counts = torch.bincount(batch, minlength=size)
        empty = (counts == 0).unsqueeze(-1)
        return torch.where(empty, torch.zeros_like(out), out)


class AttentionPooling(nn.Module):
    """Learned soft attention over nodes within each graph."""

    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        self.out_dim = dim
        gate_hidden = hidden_dim or dim
        self.gate = nn.Sequential(
            nn.Linear(dim, gate_hidden), nn.Tanh(), nn.Linear(gate_hidden, 1)
        )

    def forward(self, h: Tensor, batch: Tensor, size: int) -> Tensor:
        scores = self.gate(h).squeeze(-1)  # [N]
        maxes = torch.full((size,), torch.finfo(scores.dtype).min, device=h.device, dtype=scores.dtype)
        maxes.scatter_reduce_(0, batch, scores, reduce="amax", include_self=True)
        w = torch.exp(scores - maxes[batch])
        denom = torch.zeros(size, device=h.device, dtype=h.dtype)
        denom.index_add_(0, batch, w)
        attn = w / denom[batch].clamp(min=1e-12)
        out = torch.zeros(size, h.shape[1], device=h.device, dtype=h.dtype)
        out.index_add_(0, batch, h * attn.unsqueeze(-1))
        return out


class SelectedNodePooling(nn.Module):
    """Read out a configurable subset of node embeddings (spec §8).

    Assumes every graph in the batch has exactly ``num_nodes`` nodes and that
    batching preserves node ordering (true for :func:`collate_graphs`).
    """

    def __init__(self, dim: int, indices: list[int], num_nodes: int):
        super().__init__()
        if not indices:
            raise ValueError("SelectedNodePooling needs at least one index")
        if min(indices) < 0 or max(indices) >= num_nodes:
            raise ValueError(f"selected indices out of range for num_nodes={num_nodes}")
        self.indices = list(indices)
        self.num_nodes = num_nodes
        self.out_dim = dim * len(self.indices)
        self.register_buffer("idx", torch.tensor(self.indices, dtype=torch.long))

    def forward(self, h: Tensor, batch: Tensor, size: int) -> Tensor:
        counts = torch.bincount(batch, minlength=size)
        if not bool((counts == self.num_nodes).all()):
            raise ValueError("SelectedNodePooling requires uniform node counts per graph")
        base = torch.arange(size, device=h.device) * self.num_nodes
        sel = (base.unsqueeze(1) + self.idx.unsqueeze(0)).reshape(-1)
        return h[sel].reshape(size, len(self.indices) * h.shape[1])


def make_pooling(
    kind: str,
    dim: int,
    num_nodes: int | None = None,
    selected_nodes: list[int] | None = None,
) -> nn.Module:
    kind = (kind or "mean").lower()
    if kind == "mean":
        return MeanPooling(dim)
    if kind == "max":
        return MaxPooling(dim)
    if kind == "attention":
        return AttentionPooling(dim)
    if kind == "selected":
        if num_nodes is None or not selected_nodes:
            raise ValueError("selected pooling requires num_nodes and selected_nodes")
        return SelectedNodePooling(dim, selected_nodes, num_nodes)
    raise ValueError(f"unknown pooling {kind!r}")


class PredictionHead(nn.Module):
    """Readout / policy head (spec §2). Linear or one-hidden-layer MLP."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
