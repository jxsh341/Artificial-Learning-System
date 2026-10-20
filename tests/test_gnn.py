"""GNN tests (spec §26): forward pass, gradients, batching, permutation."""
from __future__ import annotations

import pytest
import torch

from artificial_system.data.loaders import collate_graphs
from artificial_system.data.schema import GraphData
from artificial_system.models.gnn import GraphEncoder
from artificial_system.models.readout import (
    AttentionPooling,
    MeanPooling,
    SelectedNodePooling,
)


def make_encoder(**kw) -> GraphEncoder:
    defaults = dict(in_dim=4, hidden_dim=16, layers=2)
    defaults.update(kw)
    return GraphEncoder(**defaults)


class TestForward:
    def test_output_shape(self, small_graph):
        enc = make_encoder()
        h = enc(small_graph)
        assert h.shape == (small_graph.num_nodes, 16)

    def test_gradients_flow(self, small_graph):
        enc = make_encoder()
        h = enc(small_graph)
        h.sum().backward()
        grads = [p.grad for p in enc.parameters() if p.requires_grad]
        assert all(g is not None for g in grads)
        assert any(float(g.abs().sum()) > 0 for g in grads)

    def test_no_edges_graph_is_finite(self):
        g = GraphData(node_features=torch.randn(5, 4), edge_index=torch.zeros(2, 0, dtype=torch.long))
        h = make_encoder()(g)
        assert torch.isfinite(h).all()

    @pytest.mark.parametrize("aggregation", ["mean", "sum", "max"])
    def test_aggregations(self, small_graph, aggregation):
        h = make_encoder(aggregation=aggregation)(small_graph)
        assert torch.isfinite(h).all()


class TestBatching:
    def test_batched_graphs_do_not_interfere(self, small_graph):
        g2 = GraphData(
            node_features=torch.randn(6, 4) * 10,
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        )
        enc = make_encoder()
        enc.eval()
        single = enc(small_graph)
        batched = enc(collate_graphs([small_graph, g2]))
        assert torch.allclose(single, batched[: small_graph.num_nodes], atol=1e-5)


class TestPermutation:
    def test_mean_pooling_permutation_invariant(self, small_graph):
        n = small_graph.num_nodes
        perm = torch.randperm(n)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(n)
        g_perm = GraphData(
            node_features=small_graph.node_features[perm],
            edge_index=inv[small_graph.edge_index],
        )
        enc = make_encoder()
        enc.eval()
        pool = MeanPooling(16)
        z1 = pool(enc(small_graph), torch.zeros(n, dtype=torch.long), 1)
        z2 = pool(enc(g_perm), torch.zeros(n, dtype=torch.long), 1)
        assert torch.allclose(z1, z2, atol=1e-5)


class TestPooling:
    def test_attention_pooling_shape_and_grads(self, small_graph):
        torch.manual_seed(0)
        pool = AttentionPooling(16)
        n = small_graph.num_nodes
        h = make_encoder()(small_graph)
        z = pool(h, torch.zeros(n, dtype=torch.long), 1)
        assert z.shape == (1, 16)
        z.sum().backward()
        assert any(float(p.grad.abs().sum()) > 0 for p in pool.parameters())

    def test_selected_node_pooling(self, small_graph):
        n = small_graph.num_nodes
        pool = SelectedNodePooling(16, indices=[0, 3], num_nodes=n)
        h = torch.arange(n * 16, dtype=torch.float32).reshape(n, 16)
        z = pool(h, torch.zeros(n, dtype=torch.long), 1)
        assert z.shape == (1, 32)
        assert torch.equal(z[0, :16], h[0])
        assert torch.equal(z[0, 16:], h[3])

    def test_pooling_respects_batch_vector(self):
        pool = MeanPooling(4)
        h = torch.ones(6, 4)
        batch = torch.tensor([0, 0, 0, 1, 1, 1])
        z = pool(h, batch, 2)
        assert z.shape == (2, 4)
        assert torch.allclose(z, torch.ones(2, 4))
