"""Ablation matrix (spec §20).

| Model | Graph computation  | Temporal computation | Purpose                        |
|-------|--------------------|----------------------|--------------------------------|
| M1    | none (raw pooling) | none                 | simple baseline                |
| M2    | GNN                | none                 | graph contribution             |
| M3    | none (raw pooling) | SSM                  | temporal contribution          |
| M4    | GNN                | SSM                  | combined system (= build_model)|
| M5    | GNN                | Transformer          | temporal architecture comparison|
| M6    | per-node MLP       | SSM                  | test whether structure matters |

All variants are assembled through the same :class:`GraphTemporalModel`
composition interface. Parameter counts are not exactly matched but kept in
the same range by reusing the configured hidden sizes.

spec §21 additionally lists architecture sweeps — those are expressed as
config overrides, not code changes (one variable at a time).
"""
from __future__ import annotations

from torch import nn

from artificial_system.models.gnn import GraphEncoder, IdentityEncoder, MLPEncoder
from artificial_system.models.gnn_ssm import GraphTemporalModel, build_model
from artificial_system.models.readout import PredictionHead, make_pooling
from artificial_system.models.ssm import NullTemporal, StateSpaceModel, TransformerTemporal

ABLATION_NAMES = ("M1", "M2", "M3", "M4", "M5", "M6")


def _graph_encoder(kind: str, config: dict, node_feature_dim: int, edge_feature_dim: int) -> nn.Module:
    gnn_cfg = config.get("gnn", {})
    hidden = int(gnn_cfg.get("hidden_dim", 64))
    if kind == "identity":
        return IdentityEncoder(node_feature_dim)
    if kind == "mlp":
        return MLPEncoder(
            node_feature_dim,
            hidden_dim=hidden,
            layers=int(gnn_cfg.get("layers", 2)),
            dropout=float(gnn_cfg.get("dropout", 0.0)),
        )
    if kind == "gnn":
        edge_dim = edge_feature_dim if gnn_cfg.get("use_edge_features", False) else 0
        return GraphEncoder(
            in_dim=node_feature_dim,
            hidden_dim=hidden,
            layers=int(gnn_cfg.get("layers", 2)),
            edge_dim=edge_dim,
            aggregation=gnn_cfg.get("aggregation", "mean"),
            residual=bool(gnn_cfg.get("residual", True)),
            dropout=float(gnn_cfg.get("dropout", 0.0)),
        )
    raise ValueError(kind)


def _temporal_module(kind: str, config: dict, input_dim: int) -> nn.Module:
    ssm_cfg = config.get("ssm", {})
    if kind == "none":
        return NullTemporal()
    if kind == "ssm":
        return StateSpaceModel(
            input_dim=input_dim,
            state_dim=int(ssm_cfg.get("state_dim", 64)),
            diagonal=bool(ssm_cfg.get("diagonal", True)),
            max_radius=float(ssm_cfg.get("max_radius", 0.99)),
            init_decay=float(ssm_cfg.get("init_decay", 0.9)),
            nonlinearity=ssm_cfg.get("nonlinearity", "tanh"),
        )
    if kind == "transformer":
        tr_cfg = config.get("transformer", {})
        return TransformerTemporal(
            input_dim=input_dim,
            d_model=int(tr_cfg.get("d_model", 64)),
            nhead=int(tr_cfg.get("nhead", 4)),
            num_layers=int(tr_cfg.get("layers", 2)),
        )
    raise ValueError(kind)


def build_ablation_model(
    name: str,
    config: dict,
    node_feature_dim: int,
    edge_feature_dim: int = 0,
    num_actions: int = 2,
    num_nodes: int | None = None,
) -> GraphTemporalModel:
    """Assemble one row of the ablation matrix."""
    if name not in ABLATION_NAMES:
        raise ValueError(f"unknown ablation {name!r}; choose from {ABLATION_NAMES}")
    if name == "M4":
        return build_model(config, node_feature_dim, edge_feature_dim, num_actions, num_nodes)

    ro_cfg = config.get("readout", {})
    enc_kind = {"M1": "identity", "M2": "gnn", "M3": "identity", "M5": "gnn", "M6": "mlp"}[name]
    tmp_kind = {"M1": "none", "M2": "none", "M3": "ssm", "M5": "transformer", "M6": "ssm"}[name]

    encoder = _graph_encoder(enc_kind, config, node_feature_dim, edge_feature_dim)
    pooling = make_pooling(
        ro_cfg.get("pooling", "mean"),
        encoder.hidden_dim,
        num_nodes=num_nodes,
        selected_nodes=ro_cfg.get("selected_nodes"),
    )
    temporal = _temporal_module(tmp_kind, config, pooling.out_dim)
    out_dim = getattr(temporal, "output_dim", pooling.out_dim)
    head = PredictionHead(out_dim, num_actions, hidden_dim=int(ro_cfg.get("head_hidden", 0)))
    return GraphTemporalModel(encoder, pooling, temporal, head)
