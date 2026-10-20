"""Environment interface (spec §12).

The environment is independent of the data source: observations arrive as
:class:`GraphData` and rewards follow the configured task, nothing else.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor

from artificial_system.data.schema import GraphData


class Environment(ABC):
    """Minimal episode interface: observation -> action -> reward -> next."""

    @property
    @abstractmethod
    def num_actions(self) -> int: ...

    @abstractmethod
    def reset(self, batch_size: int) -> GraphData:
        """Start a new episode for ``batch_size`` parallel instances."""

    @abstractmethod
    def current_targets(self) -> Tensor:
        """Target action per instance; ``-100`` where no response is required."""

    @abstractmethod
    def step(self, actions: Tensor) -> tuple[GraphData, Tensor, Tensor, dict]:
        """Apply actions; return ``(observation, reward, done, info)``."""
