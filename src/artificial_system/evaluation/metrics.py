"""Core evaluation metrics (spec §14-§17)."""
from __future__ import annotations

import numpy as np


def curve_endpoints(values: np.ndarray, frac: float = 0.10) -> tuple[float, float]:
    """Mean over the first and last ``frac`` of a curve (acquisition §14)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    k = max(1, int(round(values.size * frac)))
    return float(np.mean(values[:k])), float(np.mean(values[-k:]))


def trials_to_criterion(
    accuracies: np.ndarray, threshold: float = 0.9, window: int = 20
) -> int:
    """First step whose trailing-``window`` mean accuracy reaches ``threshold``.

    Returns the 1-indexed step count, or -1 if the criterion is never met.
    """
    acc = np.asarray(accuracies, dtype=float)
    acc = acc[~np.isnan(acc)]
    n = acc.size
    for i in range(n):
        lo = max(0, i - window + 1)
        if acc[lo : i + 1].mean() >= threshold:
            return i + 1
    return -1


def retention_ratio(post_delay_accuracy: float, post_training_accuracy: float) -> float:
    """retention = post-delay performance / post-training performance (§15)."""
    if post_training_accuracy <= 0:
        return float("nan")
    return float(post_delay_accuracy / post_training_accuracy)


def residual_preference(
    actions: np.ndarray, cues: np.ndarray, old_mapping: np.ndarray
) -> float:
    """Fraction of actions still matching the *old* mapping (§16)."""
    actions = np.asarray(actions)
    cues = np.asarray(cues)
    old_mapping = np.asarray(old_mapping)
    if actions.size == 0:
        return float("nan")
    return float(np.mean(actions == old_mapping[cues]))


def internal_state_stats(states: np.ndarray) -> dict:
    """Summary statistics over a set of hidden states ``[T, B, H]`` (§18)."""
    states = np.asarray(states, dtype=float)
    norms = np.linalg.norm(states, axis=-1)
    flat = states.reshape(-1, states.shape[-1])
    if flat.shape[0] > 1:
        sims = flat @ flat.T
        norms_flat = np.linalg.norm(flat, axis=1, keepdims=True)
        cos = sims / (norms_flat @ norms_flat.T + 1e-8)
        iu = np.triu_indices_from(cos, k=1)
        mean_cos = float(np.mean(cos[iu]))
    else:
        mean_cos = float("nan")
    return {
        "state_norm_mean": float(np.mean(norms)),
        "state_norm_std": float(np.std(norms)),
        "mean_pairwise_cosine": mean_cos,
    }
