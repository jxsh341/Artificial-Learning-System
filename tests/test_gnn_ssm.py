"""Combined model tests (spec §26): graph -> GNN -> SSM -> output,
in both batched and single-step execution."""
from __future__ import annotations

import torch

from artificial_system.data.loaders import collate_graphs
from artificial_system.data.schema import GraphData
from artificial_system.models import build_model


def make_sequence(graph: GraphData, t: int, batch_size: int) -> list[GraphData]:
    torch.manual_seed(1)
    seq = []
    for _ in range(t):
        g = GraphData(
            node_features=graph.node_features + 0.01 * torch.randn_like(graph.node_features),
            edge_index=graph.edge_index,
        )
        seq.append(collate_graphs([g] * batch_size))
    return seq


class TestCombinedModel:
    def test_forward_shapes(self, cfg, small_graph):
        model = build_model(cfg, node_feature_dim=4, num_actions=2, num_nodes=small_graph.num_nodes)
        seq = make_sequence(small_graph, t=5, batch_size=3)
        logits, state, _ = model(seq)
        assert logits.shape == (5, 3, 2)
        assert state.shape == (3, 16)

    def test_step_matches_forward(self, cfg, small_graph):
        torch.manual_seed(0)
        model = build_model(cfg, node_feature_dim=4, num_actions=2, num_nodes=small_graph.num_nodes)
        seq = make_sequence(small_graph, t=4, batch_size=2)
        logits_seq, state_seq, _ = model(seq)
        state = None
        steps = []
        for g in seq:
            logits, state, _ = model.step(g, state)
            steps.append(logits)
        assert torch.allclose(logits_seq, torch.stack(steps, 0), atol=1e-5)
        assert torch.allclose(state_seq, state, atol=1e-5)

    def test_single_step_online_interface(self, cfg, small_graph):
        model = build_model(cfg, node_feature_dim=4, num_actions=2, num_nodes=small_graph.num_nodes)
        state = model.initial_state(2)
        batched = collate_graphs([small_graph, small_graph])
        logits, state, aux = model.step(batched, state, return_aux=True)
        assert logits.shape == (2, 2)
        assert aux["pooled"].shape == (2, 16)
        assert aux["node_embeddings"].shape == (2 * small_graph.num_nodes, 16)

    def test_gradient_flows_to_gnn_and_ssm(self, cfg, small_graph):
        model = build_model(cfg, node_feature_dim=4, num_actions=2, num_nodes=small_graph.num_nodes)
        seq = make_sequence(small_graph, t=3, batch_size=2)
        logits, _, _ = model(seq)
        logits.sum().backward()
        gnn_grads = [p.grad for p in model.graph_encoder.parameters()]
        ssm_grads = [p.grad for p in model.temporal_model.parameters()]
        assert any(g is not None and float(g.abs().sum()) > 0 for g in gnn_grads)
        assert any(g is not None and float(g.abs().sum()) > 0 for g in ssm_grads)

    def test_attention_pooling_variant(self, cfg, small_graph):
        cfg = {**cfg, "readout": {"pooling": "attention"}}
        model = build_model(cfg, node_feature_dim=4, num_actions=2, num_nodes=small_graph.num_nodes)
        seq = make_sequence(small_graph, t=2, batch_size=2)
        logits, _, _ = model(seq)
        assert logits.shape == (2, 2, 2)

    def test_selected_pooling_variant(self, cfg, small_graph):
        cfg = {**cfg, "readout": {"pooling": "selected", "selected_nodes": [0, 1, 2]}}
        model = build_model(cfg, node_feature_dim=4, num_actions=2, num_nodes=small_graph.num_nodes)
        seq = make_sequence(small_graph, t=2, batch_size=2)
        logits, _, _ = model(seq)
        assert logits.shape == (2, 2, 2)
