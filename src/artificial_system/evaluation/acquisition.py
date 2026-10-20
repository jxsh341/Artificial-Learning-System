"""Acquisition experiment (spec §14).

Train from a (freshly seeded) model and measure how performance changes
with repeated experience: performance before vs after training.
"""
from __future__ import annotations

import numpy as np

from artificial_system.evaluation.metrics import curve_endpoints, trials_to_criterion
from artificial_system.training.trainer import Trainer


def run_acquisition(
    trainer: Trainer,
    episodes: int,
    criterion: float = 0.9,
    criterion_window: int = 25,
) -> tuple[list[dict], dict]:
    """Train and summarise the acquisition curve.

    Returns ``(per_step_records, summary)``.
    """
    records = trainer.train(episodes)
    accs = np.array([r.get("accuracy", np.nan) for r in records], dtype=float)
    active = ~np.isnan(accs)
    acc_active = accs[active]
    first, last = curve_endpoints(acc_active)
    rewards = np.array([r.get("reward", np.nan) for r in records], dtype=float)[active]
    _, reward_last = curve_endpoints(rewards)
    summary = {
        "steps": len(records),
        "active_steps": int(active.sum()),
        "accuracy_first_10pct": first,
        "accuracy_last_10pct": last,
        "accuracy_delta": last - first,
        "trials_to_criterion": trials_to_criterion(
            acc_active, threshold=criterion, window=criterion_window
        ),
        "reward_last_10pct": reward_last,
    }
    return records, summary
