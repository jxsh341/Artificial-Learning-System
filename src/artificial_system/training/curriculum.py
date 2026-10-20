"""Simple curriculum utilities (spec: experiment-first, minimal scalars)."""
from __future__ import annotations


class LinearCurriculum:
    """Linearly ramp a scalar parameter from ``start`` to ``stop``.

    Example: gradually increase ``delay`` or the number of active cues.
    """

    def __init__(self, start: float, stop: float, num_steps: int):
        self.start = float(start)
        self.stop = float(stop)
        self.num_steps = max(1, int(num_steps))

    def value(self, step: int) -> float:
        frac = min(max(step / self.num_steps, 0.0), 1.0)
        return self.start + frac * (self.stop - self.start)


def make_curriculum(config: dict | None) -> LinearCurriculum | None:
    if not config:
        return None
    return LinearCurriculum(
        start=config.get("start", 0.0),
        stop=config.get("stop", 0.0),
        num_steps=config.get("num_steps", 1),
    )
