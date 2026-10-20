"""Combined GNN + SSM model (spec §11).

    graph state at t -> GNN -> pooling -> z_t -> SSM -> temporal state
        -> readout head -> prediction/action

Supports both ``forward(sequence_of_graph_states)`` and the online
``step(graph_state, recurrent_state)`` interface.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from artificial_system.data.schema import GraphData
from artificial_system.models.gnn import GraphEncoder
from artificial_system.models.readout import PredictionHead, make_pooling
from artificial_system.models.ssm import StateSpaceModel


class GraphTemporalModel(nn.Module):
    """Composable graph-temporal model.

    Any module with ``forward(GraphData) -> [N, H]`` can serve as the graph
    encoder, and any module exposing ``initial_state / step(z, state)`` can
    serve as the temporal model — this is what keeps the ablation matrix
    (spec §20) cheap.
    """

    def __init__(
        self,
        graph_encoder: nn.Module,
        pooling: nn.Module,
        temporal_model: nn.Module,
        head: nn.Module,
    ):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.pooling = pooling
        self.temporal_model = temporal_model
        self.head = head

    def encode(self, graph: GraphData) -> tuple[Tensor, Tensor]:
        """GraphData -> (pooled [B, H], node embeddings [N, H])."""
        h = self.graph_encoder(graph)
        if h.device != graph.node_features.device:
            h = h.to(graph.node_features.device)
        batch = graph.batch
        if batch is None:
            batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=h.device)
        z = self.pooling(h, batch, graph.num_graphs)
        return z, h

    def initial_state(self, batch_size: int, device=None, dtype=None):
        return self.temporal_model.initial_state(batch_size, device, dtype)

    def reset_state(self, state, mask=None):
        """Full or partial SSM state reset (spec §19 temporal perturbations)."""
        return self.temporal_model.reset_state(state, mask)

    def step(
        self,
        graph: GraphData,
        state=None,
        return_aux: bool = False,
    ) -> tuple[Tensor, object, dict]:
        """One online step. Returns ``(logits, new_state, aux)``."""
        z, h = self.encode(graph)
        y, state = self.temporal_model.step(z, state)
        logits = self.head(y)
        aux: dict = {}
        if return_aux:
            aux = {
                "pooled": z,
                "hidden": y,
                "node_embeddings": h,
                "state": state if isinstance(state, Tensor) else None,
            }
        return logits, state, aux

    def forward(
        self,
        graph_sequence: list[GraphData],
        initial_state=None,
        return_aux: bool = False,
    ) -> tuple[Tensor, object, list[dict]]:
        """Process a sequence of graph states -> logits ``[T, B, num_actions]``."""
        state = initial_state
        all_logits, all_aux = [], []
        for graph in graph_sequence:
            logits, state, aux = self.step(graph, state, return_aux=return_aux)
            all_logits.append(logits)
            if return_aux:
                all_aux.append(aux)
        return torch.stack(all_logits, dim=0), state, all_aux


def build_model(
    config: dict,
    node_feature_dim: int,
    edge_feature_dim: int = 0,
    num_actions: int = 2,
    num_nodes: int | None = None,
) -> GraphTemporalModel:
    """Assemble the combined model from a configuration dictionary."""
    gnn_cfg = config.get("gnn", {})
    ssm_cfg = config.get("ssm", {})
    ro_cfg = config.get("readout", {})

    hidden = int(gnn_cfg.get("hidden_dim", 64))
    edge_dim = edge_feature_dim if gnn_cfg.get("use_edge_features", False) else 0
    encoder = GraphEncoder(
        in_dim=node_feature_dim,
        hidden_dim=hidden,
        layers=int(gnn_cfg.get("layers", 2)),
        edge_dim=edge_dim,
        aggregation=gnn_cfg.get("aggregation", "mean"),
        residual=bool(gnn_cfg.get("residual", True)),
        dropout=float(gnn_cfg.get("dropout", 0.0)),
    )
    pooling = make_pooling(
        ro_cfg.get("pooling", "mean"),
        hidden,
        num_nodes=num_nodes,
        selected_nodes=ro_cfg.get("selected_nodes"),
    )
    temporal = StateSpaceModel(
        input_dim=pooling.out_dim,
        state_dim=int(ssm_cfg.get("state_dim", 64)),
        output_dim=int(ssm_cfg.get("output_dim", ssm_cfg.get("state_dim", 64))),
        diagonal=bool(ssm_cfg.get("diagonal", True)),
        max_radius=float(ssm_cfg.get("max_radius", 0.99)),
        init_decay=float(ssm_cfg.get("init_decay", 0.9)),
        nonlinearity=ssm_cfg.get("nonlinearity", "tanh"),
    )
    head = PredictionHead(
        temporal.output_dim,
        num_actions,
        hidden_dim=int(ro_cfg.get("head_hidden", 0)),
        dropout=float(ro_cfg.get("head_dropout", 0.0)),
    )
    return GraphTemporalModel(encoder, pooling, temporal, head)
