"""Task construction helpers (spec §12, §17).

Cue feature sets with explicit similarity structure so that generalization
can be measured against a known relationship rather than identity matching.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def make_cue_features(
    num_cues: int,
    cue_dim: int,
    structure: str = "onehot",
    num_families: int = 2,
    feature_noise: float = 0.25,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """Create the cue feature prototypes.

    Returns ``(features [num_cues, cue_dim], family_ids [num_cues])``.

    - ``onehot``: each cue is a one-hot axis (padded with zeros), one family
      per cue.
    - ``hierarchical``: ``num_families`` orthogonal-ish prototypes; cues are
      noisy samples around their family prototype, giving an explicit
      similarity structure for generalization tests.
    """
    rng = np.random.default_rng(seed)
    if structure == "onehot":
        if cue_dim < num_cues:
            raise ValueError("onehot cue structure requires cue_dim >= num_cues")
        feats = np.zeros((num_cues, cue_dim), dtype=np.float32)
        feats[np.arange(num_cues), np.arange(num_cues)] = 1.0
        family_ids = np.arange(num_cues)
    elif structure == "hierarchical":
        num_families = max(1, min(num_families, num_cues))
        prototypes = rng.standard_normal((num_families, cue_dim)).astype(np.float32)
        prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8
        family_ids = np.array([i % num_families for i in range(num_cues)])
        noise = rng.standard_normal((num_cues, cue_dim)).astype(np.float32) * feature_noise
        feats = 2.0 * prototypes[family_ids] + noise
    else:
        raise ValueError(f"unknown cue structure {structure!r}")
    return (
        torch.from_numpy(feats.astype(np.float32)),
        torch.from_numpy(np.asarray(family_ids, dtype=np.int64)),
    )


def make_mapping(
    num_cues: int,
    num_actions: int,
    mapping="cyclic",
    family_ids: Tensor | None = None,
) -> Tensor:
    """Default cue -> action mappings.

    - ``cyclic``: ``action = cue % num_actions``.
    - ``family``: ``action = family_id % num_actions`` (ties actions to the
      stimulus similarity structure).
    - explicit sequence of per-cue action ids.
    """
    if isinstance(mapping, str):
        if mapping == "cyclic":
            return torch.arange(num_cues) % num_actions
        if mapping == "family":
            if family_ids is None:
                raise ValueError("family mapping requires family_ids")
            return family_ids % num_actions
        raise ValueError(f"unknown mapping {mapping!r}")
    m = torch.as_tensor(list(mapping), dtype=torch.long)
    if m.shape[0] != num_cues:
        raise ValueError(f"mapping must have {num_cues} entries")
    if int(m.min()) < 0 or int(m.max()) >= num_actions:
        raise ValueError("mapping contains invalid action id")
    return m


def reversed_mapping(mapping: Tensor, num_actions: int) -> Tensor:
    """Default reversal: shift every target action by one (spec §16)."""
    return (mapping + 1) % num_actions
