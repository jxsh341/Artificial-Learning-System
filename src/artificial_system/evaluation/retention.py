"""Retention experiment (spec §15).

After training: stop learning, reset the environment, wait a configurable
number of blank/distractor steps (letting the SSM state evolve without
feedback), then probe the same cue -> action mapping across several delays.
The SSM hidden state is recorded for internal-state analysis (§18).
"""
from __future__ import annotations

import numpy as np
import torch

from artificial_system.evaluation.metrics import internal_state_stats, retention_ratio


@torch.no_grad()
def probe_mapping(
    model,
    env,
    num_steps: int,
    state=None,
) -> tuple[float, np.ndarray]:
    """Measure accuracy on the current mapping without any learning.

    Returns ``(accuracy, hidden_states [T, B, H])`` (empty states for
    stateless temporal modules).
    """
    model.eval()
    correct, total = 0, 0
    states = []
    batch = env.batch_size
    for _ in range(num_steps):
        cues = env.sample_cues(batch)
        obs = env.cue_observation(cues)
        logits, state, _ = model.step(obs, state)
        targets = env.mapping_tensor()[cues]
        pred = logits.argmax(-1)
        correct += int((pred == targets).sum().item())
        total += targets.numel()
        if isinstance(state, torch.Tensor):
            states.append(state.clone())
    model.train()
    acc = correct / max(total, 1)
    stacked = torch.stack(states, dim=0).numpy() if states else np.zeros((0, batch, 0))
    return acc, stacked


@torch.no_grad()
def run_retention(
    model,
    env,
    delays: list[int],
    test_steps: int,
    distractor: str = "blank",
) -> tuple[list[dict], dict]:
    """Run the delay sweep. Returns ``(rows, summary)``."""
    model.eval()
    batch = env.batch_size
    rows: list[dict] = []
    for delay in delays:
        env.reset(batch)  # reset environment; no learning feedback anywhere here
        state = model.initial_state(batch)
        for _ in range(int(delay)):
            if distractor == "cues":
                obs = env.cue_observation(env.sample_cues(batch))
            else:
                obs = env.blank_observation()
            _, state, _ = model.step(obs, state)
        acc, states = probe_mapping(model, env, test_steps, state)
        row = {
            "delay": int(delay),
            "accuracy": float(acc),
            "retention_ratio": float("nan"),  # filled below
            "test_steps": int(test_steps),
        }
        row.update(internal_state_stats(states) if states.shape[0] else {})
        rows.append(row)
    base = rows[0]["accuracy"] if rows else float("nan")
    for row in rows:
        row["retention_ratio"] = retention_ratio(row["accuracy"], base)
    summary = {f"retention_ratio@{row['delay']}": row["retention_ratio"] for row in rows}
    summary["retention_base_accuracy"] = base
    model.train()
    return rows, summary
