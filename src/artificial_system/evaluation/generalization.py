"""Generalization experiment (spec §17).

Train on a subset of cues, then probe held-out cues whose correct response
is defined by the same underlying rule. Performance is reported overall and
per stimulus family to test whether transfer follows the explicit
similarity structure.
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def probe_cue_set(
    model,
    env_factory,
    allowed_cues: list[int],
    num_steps: int,
    seed: int,
    batch_size: int = 16,
) -> dict:
    """Probe accuracy on a restricted cue set without any learning."""
    env = env_factory(allowed_cues=allowed_cues, seed=seed)
    env.reset(batch_size)
    state = model.initial_state(batch_size)
    correct, total = 0, 0
    per_family: dict[int, list[int]] = {}
    model.eval()
    for _ in range(num_steps):
        cues = env.sample_cues(batch_size)
        obs = env.cue_observation(cues)
        logits, state, _ = model.step(obs, state)
        targets = env.mapping_tensor()[cues]
        pred = logits.argmax(-1)
        hit = pred == targets
        correct += int(hit.sum().item())
        total += targets.numel()
        if env.family_ids is not None:
            fams = env.family_ids[cues]
            for h, f in zip(hit.tolist(), fams.tolist()):
                per_family.setdefault(int(f), []).append(int(h))
    model.train()
    return {
        "accuracy": correct / max(total, 1),
        "per_family": {str(k): float(np.mean(v)) for k, v in per_family.items()},
        "num_cues": len(allowed_cues),
    }


def run_generalization(
    model,
    env_factory,
    train_cues: list[int],
    test_cues: list[int],
    test_steps: int,
    seed: int = 999,
    batch_size: int = 16,
) -> tuple[list[dict], dict]:
    """Compare performance on trained vs held-out cues."""
    train_probe = probe_cue_set(model, env_factory, train_cues, test_steps, seed, batch_size)
    test_probe = probe_cue_set(model, env_factory, test_cues, test_steps, seed + 1, batch_size)
    rows = [
        {"split": "train", "accuracy": train_probe["accuracy"], "num_cues": len(train_cues)},
        {"split": "test", "accuracy": test_probe["accuracy"], "num_cues": len(test_cues)},
    ]
    for fam, acc in test_probe["per_family"].items():
        rows.append({"split": f"test_family_{fam}", "accuracy": acc, "num_cues": len(test_cues)})
    summary = {
        "generalization_train_accuracy": train_probe["accuracy"],
        "generalization_test_accuracy": test_probe["accuracy"],
        "generalization_transfer_gap": train_probe["accuracy"] - test_probe["accuracy"],
    }
    return rows, summary
