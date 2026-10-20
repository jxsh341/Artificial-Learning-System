"""Graph data adapters (spec §5, §28 — the integration boundary).

The production graph source plugs in here. Implement a new
:class:`GraphAdapter` (or load an ``.npz`` export) and everything upstream
(GNN, SSM, training, evaluation) stays unchanged.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

from artificial_system.data.graph import FAMILY_NAMES, SyntheticGraphGenerator
from artificial_system.data.schema import GraphData


class GraphAdapter(ABC):
    """Anything that can produce a GraphData from a data source."""

    @abstractmethod
    def load(self) -> GraphData:
        """Load, validate, and return the graph."""


class SyntheticAdapter(GraphAdapter):
    """Adapter over the built-in synthetic graph families."""

    def __init__(self, family: str = "random", **graph_config):
        self.generator = SyntheticGraphGenerator(
            {**graph_config, "family": family}, family=family
        )

    def load(self) -> GraphData:
        return self.generator.generate().validate()


class NpzAdapter(GraphAdapter):
    """Adapter for ``.npz`` exports.

    Expected keys: ``node_features`` ``[N, F]``, ``edge_index`` ``[2, E]``;
    optional: ``edge_features`` ``[E, Fe]``, ``node_mask`` ``[N]``.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> GraphData:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        data = np.load(self.path)
        graph = GraphData(
            node_features=torch.from_numpy(np.asarray(data["node_features"], dtype=np.float32)),
            edge_index=torch.from_numpy(np.asarray(data["edge_index"], dtype=np.int64)),
            edge_features=(
                torch.from_numpy(np.asarray(data["edge_features"], dtype=np.float32))
                if "edge_features" in data
                else None
            ),
            node_mask=(
                torch.from_numpy(np.asarray(data["node_mask"], dtype=bool))
                if "node_mask" in data
                else None
            ),
            metadata={"adapter": "npz", "path": str(self.path)},
        )
        return graph.validate()


_REGISTRY: dict[str, type[GraphAdapter]] = {
    "synthetic": SyntheticAdapter,
    "npz": NpzAdapter,
}


def register_adapter(name: str, adapter_cls: type[GraphAdapter]) -> None:
    """Register a custom adapter (e.g. for the production graph source)."""
    _REGISTRY[name] = adapter_cls


def get_adapter(name: str, **kwargs) -> GraphAdapter:
    if name not in _REGISTRY:
        raise KeyError(f"unknown adapter {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def load_graph(source: str, graph_config: dict) -> GraphData:
    """Resolve a graph source string to a validated GraphData.

    ``source`` may be a synthetic family name (``random``, ``regular``,
    ``small_world``, ``scale_free``, ``modular``) or a path to an ``.npz``
    file (optionally prefixed with ``npz:``).
    """
    cfg = dict(graph_config)
    cfg.pop("family", None)
    if source in FAMILY_NAMES:
        return SyntheticAdapter(family=source, **cfg).load()
    path = source[len("npz:") :] if source.startswith("npz:") else source
    if path.endswith(".npz"):
        return NpzAdapter(path).load()
    raise ValueError(
        f"unrecognised graph source {source!r}; "
        f"use one of {FAMILY_NAMES} or a path to an .npz export"
    )
