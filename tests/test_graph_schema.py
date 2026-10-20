"""Graph interface tests (spec §26): dimensions, indexing, batching, features."""
from __future__ import annotations

import pytest
import torch

from artificial_system.data.graph import FAMILY_NAMES, SyntheticGraphGenerator
from artificial_system.data.loaders import collate_graphs, unbatch_graph
from artificial_system.data.schema import GraphData


def make_graph(n=6, f=3, edges=None, edge_dim=0, seed=0) -> GraphData:
    g = torch.Generator().manual_seed(seed)
    if edges is None:
        edges = [(i, (i + 1) % n) for i in range(n)]
    return GraphData(
        node_features=torch.randn(n, f, generator=g),
        edge_index=torch.tensor(edges, dtype=torch.long).t(),
        edge_features=torch.randn(len(edges), edge_dim, generator=g) if edge_dim else None,
    )


class TestSchemaValidation:
    def test_valid_graph_passes(self, small_graph):
        assert small_graph.validate() is small_graph

    def test_bad_edge_index_shape_raises(self, small_graph):
        bad = GraphData(small_graph.node_features, torch.zeros(3, 4, dtype=torch.long))
        with pytest.raises(ValueError):
            bad.validate()

    def test_edge_index_out_of_range_raises(self, small_graph):
        bad = GraphData(
            small_graph.node_features,
            torch.tensor([[0, 99], [1, 2]], dtype=torch.long),
        )
        with pytest.raises(ValueError):
            bad.validate()

    def test_edge_features_wrong_rows_raise(self, small_graph):
        bad = GraphData(
            small_graph.node_features,
            small_graph.edge_index,
            edge_features=torch.randn(3, 2),
        )
        with pytest.raises(ValueError):
            bad.validate()

    def test_bad_node_mask_raises(self, small_graph):
        bad = GraphData(
            small_graph.node_features,
            small_graph.edge_index,
            node_mask=torch.ones(3, dtype=torch.bool),
        )
        with pytest.raises(ValueError):
            bad.validate()


class TestBatching:
    def test_collate_offsets_edge_index(self):
        g1 = make_graph(n=4, edges=[(0, 1), (1, 2)])
        g2 = make_graph(n=5, edges=[(0, 1), (2, 3)])
        batched = collate_graphs([g1, g2])
        assert batched.num_nodes == 9
        assert batched.num_graphs == 2
        assert batched.batch.tolist() == [0] * 4 + [1] * 5
        assert int(batched.edge_index[:, g1.num_edges :].min()) >= 4

    def test_collate_preserves_features(self):
        g1 = make_graph(n=4, edge_dim=2, seed=1)
        g2 = make_graph(n=5, edge_dim=2, seed=2)
        batched = collate_graphs([g1, g2])
        assert torch.allclose(batched.node_features[:4], g1.node_features)
        assert torch.allclose(batched.node_features[4:], g2.node_features)
        assert torch.allclose(batched.edge_features[: g1.num_edges], g1.edge_features)

    def test_collate_rejects_mismatched_edge_features(self):
        g1 = make_graph(n=4, edge_dim=2)
        g2 = make_graph(n=4, edge_dim=0)
        with pytest.raises(ValueError):
            collate_graphs([g1, g2])

    def test_unbatch_roundtrip(self):
        g1, g2 = make_graph(n=4, seed=3), make_graph(n=4, seed=4)
        parts = unbatch_graph(collate_graphs([g1, g2]))
        assert len(parts) == 2
        assert torch.allclose(parts[0].node_features, g1.node_features)
        assert torch.equal(parts[1].edge_index, g2.edge_index)


class TestSyntheticGenerators:
    @pytest.mark.parametrize("family", FAMILY_NAMES)
    def test_all_families_generate_valid_graphs(self, family):
        gen = SyntheticGraphGenerator(
            {
                "family": family,
                "num_nodes": 40,
                "node_feature_dim": 8,
                "edge_feature_dim": 2,
                "directed": True,
                "seed": 7,
            }
        )
        g = gen.generate()
        assert g.num_nodes == 40
        assert g.node_feature_dim == 8
        assert g.num_edges > 0
        assert g.edge_features.shape == (g.num_edges, 2)
        assert g.metadata["family"] == family

    def test_reproducible_with_seed(self):
        cfg = {"family": "small_world", "num_nodes": 30, "node_feature_dim": 4, "seed": 11}
        g1 = SyntheticGraphGenerator(cfg).generate()
        g2 = SyntheticGraphGenerator(cfg).generate()
        assert torch.equal(g1.edge_index, g2.edge_index)
        assert torch.allclose(g1.node_features, g2.node_features)

    def test_undirected_adds_both_directions(self):
        g = SyntheticGraphGenerator(
            {"family": "random", "num_nodes": 20, "node_feature_dim": 4,
             "directed": False, "seed": 5, "params": {"p": 0.2}}
        ).generate()
        pairs = set(map(tuple, g.edge_index.t().tolist()))
        for u, v in pairs:
            assert (v, u) in pairs

    def test_directedness_default_produces_directed_edges(self):
        g = SyntheticGraphGenerator(
            {"family": "random", "num_nodes": 20, "node_feature_dim": 4,
             "directed": True, "seed": 6, "params": {"p": 0.2}}
        ).generate()
        g.validate()
