"""Learning objectives (spec §13).

Supervised objectives first; a simple policy-gradient loss is provided for
later reward-based experiments once the architecture trains reliably.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

IGNORE_INDEX = -100


def classification_loss(logits: Tensor, targets: Tensor, ignore_index: int = IGNORE_INDEX) -> Tensor:
    """Cross-entropy that tolerates fully-masked steps (returns a zero loss)."""
    if not bool((targets != ignore_index).any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits, targets, ignore_index=ignore_index)


def policy_gradient_loss(
    logits: Tensor,
    actions: Tensor,
    rewards: Tensor,
    baseline: Tensor | None = None,
) -> Tensor:
    """REINFORCE with optional baseline (for post-supervised phases)."""
    logp = F.log_softmax(logits, dim=-1)
    chosen = logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    advantage = rewards if baseline is None else rewards - baseline
    return -(chosen * advantage.detach()).mean()
