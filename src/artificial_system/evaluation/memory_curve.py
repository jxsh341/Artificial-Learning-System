"""Memory-retention curve (phase 2: "stress the memory").

Measures accuracy as a function of temporal distance between cue and
response. For each delay ``d``: present the cue once, insert ``d`` blank
(or distractor) steps during which the SSM state evolves without feedback,
then require a single response. Weights are frozen — the curve isolates how
long the cue's trace persists in the temporal state against ongoing input.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def run_memory_curve(
    model,
    env,
    delays: list[int],
    repeats: int = 20,
    distractor: str = "blank",
    repeats_floor: int = 3,
) -> tuple[list[dict], dict]:
    """Returns ``(rows, summary)`` with one row per delay.

    Repeats are reduced for large delays to bound total probe steps:
    ``eff = repeats`` for ``d <= 50``, else ``max(repeats_floor, repeats // (d // 50))``.
    """
    model.eval()
    batch = env.batch_size
    rows: list[dict] = []
    ceiling = None
    for d in delays:
        d = int(d)
        eff = repeats if d <= 50 else max(repeats_floor, repeats // max(1, d // 50))
        correct = total = 0
        for _ in range(eff):
            cues = env.sample_cues(batch)
            obs = env.cue_observation(cues)  # cue presentation
            _, state, _ = model.step(obs, None)
            for _ in range(d):
                if distractor == "cues":
                    obs = env.cue_observation(env.sample_cues(batch))
                else:
                    obs = env.blank_observation()
                _, state, _ = model.step(obs, state)
            logits, _, _ = model.step(env.blank_observation(), state)  # response
            pred = logits.argmax(-1)
            targets = env.mapping_tensor()[cues]
            correct += int((pred == targets).sum().item())
            total += int(targets.numel())
        acc = correct / max(total, 1)
        if ceiling is None:
            ceiling = acc
        rows.append({"delay": d, "accuracy": float(acc), "repeats": eff})

    # Half-life: largest *contiguous* prefix of delays still above
    # max(0.5 * ceiling, chance + margin) — robust to chance fluctuations
    # at long delays.
    chance = 1.0 / max(env.num_actions, 1)
    threshold = max(0.5 * ceiling, chance + 0.05) if ceiling and ceiling > 0 else chance
    half_life = 0
    for row in rows:
        if row["accuracy"] >= threshold:
            half_life = row["delay"]
        else:
            break
    summary = {f"memory_accuracy@{row['delay']}": row["accuracy"] for row in rows}
    summary["memory_ceiling"] = ceiling
    summary["memory_half_life"] = half_life
    summary["memory_half_life_threshold"] = threshold
    model.train()
    return rows, summary
