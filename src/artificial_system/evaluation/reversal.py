"""Reversal experiment (spec §16).

The model (already trained on mapping 1) must adapt after the cue -> action
mapping is flipped. Measures reversal speed (trials to criterion), residual
preference for the old mapping, and final performance.
"""
from __future__ import annotations

import numpy as np
import torch

from artificial_system.evaluation.metrics import (
    curve_endpoints,
    residual_preference,
    trials_to_criterion,
)
from artificial_system.training.trainer import Trainer


@torch.no_grad()
def measure_residual_preference(model, env, old_mapping: torch.Tensor, num_steps: int) -> float:
    """Fraction of responses matching the *old* mapping, before any relearning."""
    model.eval()
    obs = env.reset(env.batch_size)
    state = model.initial_state(env.batch_size)
    actions_all, cues_all, active_all = [], [], []
    for _ in range(num_steps):
        logits, state, _ = model.step(obs, state)
        targets = env.current_targets()
        active = targets != -100
        actions = logits.argmax(-1)
        obs, _, _, info = env.step(actions)
        if bool(active.any()):
            m = active
            actions_all.append(actions[m])
            cues_all.append(info["cues"][m])
    model.train()
    if not actions_all:
        return float("nan")
    actions = torch.cat(actions_all).numpy()
    cues = torch.cat(cues_all).numpy()
    return residual_preference(actions, cues, old_mapping.numpy())


def run_reversal(
    trainer: Trainer,
    episodes: int,
    criterion: float = 0.9,
    criterion_window: int = 25,
    residual_steps: int = 50,
) -> tuple[list[dict], dict]:
    """Reverse the mapping, retrain, and summarise.

    Assumes the trainer's model was already trained on the original mapping.
    Returns ``(per_step_records, summary)``.
    """
    env = trainer.env
    old_mapping = env.mapping_tensor().clone()
    env.reverse()
    residual = measure_residual_preference(trainer.model, env, old_mapping, residual_steps)
    records = trainer.train(episodes)
    accs = np.array([r.get("accuracy", np.nan) for r in records], dtype=float)
    acc_active = accs[~np.isnan(accs)]
    _, last = curve_endpoints(acc_active)
    summary = {
        "residual_preference_old_mapping": residual,
        "trials_to_criterion_post_reversal": trials_to_criterion(
            acc_active, threshold=criterion, window=criterion_window
        ),
        "accuracy_last_10pct_post_reversal": last,
        "reversal_steps": len(records),
    }
    return records, summary
