"""Controlled size-generalization sweep (TECHNICAL REPORT §7 item 2).

Goal: characterize the unseen-size anomaly (some successful runs stall at
chance on n=200 graphs) by evaluating *saved checkpoints* across a ladder of
graph sizes, while isolating node count from the mean-degree confound:

- ``density_mode="fixed"``: keep ``edge_density`` constant — node count AND
  mean degree both grow (this is what the standard unseen-size probe does);
- ``density_mode="degree_matched"``: scale ``edge_density`` ∝ base_n/N so the
  mean degree stays at the training-size level — only the node count changes.

For every (size, mode, independent instance set) we record behavioural
accuracy (immediate; delayed with intact state; delayed after state reset)
plus activation statistics at the presentation step (pooled-graph norm,
node-embedding norm, attention concentration, marker-node attention share),
so gradual degradation of representation scale can be distinguished from an
abrupt readout mismatch. Checkpoints are treated as data: nothing is retrained.
"""
from __future__ import annotations

import torch

from artificial_system.data.graph_tasks import make_connectivity_dataset
from artificial_system.evaluation.graph_classification import (
    _chunks,
    observe_instances,
)

GEN_DEFAULTS = dict(
    feature_dim=16,
    marker_dim=2,
    max_path_len=3,
    edge_density=0.04,
    directed=False,
    label_mode="adjacency",
    neg_min_distance=3,
)


def _gen_cfg(task_cfg: dict, graph_cfg: dict) -> dict:
    base = dict(GEN_DEFAULTS)
    for key in base:
        if key in task_cfg:
            base[key] = task_cfg[key]
    base["feature_dim"] = int(
        graph_cfg.get("node_feature_dim", base["feature_dim"])
    )
    base["marker_dim"] = int(task_cfg.get("marker_dim", base["marker_dim"]))
    return base


def _degree_matched_density(base_density: float, base_nodes: int, num_nodes: int) -> float:
    """Density that keeps the mean degree at the base-size training level."""
    return base_density * base_nodes / num_nodes


def _attention_weights(model, h: torch.Tensor, batch: torch.Tensor, size: int) -> torch.Tensor:
    """Recompute the per-node attention weights (mirrors AttentionPooling)."""
    scores = model.pooling.gate(h).squeeze(-1)
    maxes = torch.full((size,), torch.finfo(scores.dtype).min, device=h.device)
    maxes.scatter_reduce_(0, batch, scores, reduce="amax", include_self=True)
    w = torch.exp(scores - maxes[batch])
    denom = torch.zeros(size, device=h.device)
    denom.index_add_(0, batch, w)
    return w / denom[batch].clamp(min=1e-12)


@torch.no_grad()
def _probe_size(
    model,
    dataset,
    delay: int,
    batch_size: int,
    marker_dim: int,
    generic_blank=None,
) -> dict:
    """Behavioural accuracy + presentation-step activation statistics."""
    import numpy as np

    model.eval()
    correct_imm = correct_intact = correct_reset = total = 0
    z_norms: list[float] = []
    h_norms: list[float] = []
    top1: list[float] = []
    marker_share: list[float] = []
    margins: list[float] = []
    for idxs in _chunks(len(dataset), batch_size):
        labels = dataset.labels[torch.as_tensor(idxs, dtype=torch.long)]
        b = len(idxs)
        obs = observe_instances(dataset, idxs, True, marker_dim)

        # --- presentation step: activations + immediate accuracy ---
        logits, _, aux = model.step(obs, None, return_aux=True)
        pred_imm = logits.argmax(-1).cpu()
        margins.extend(
            (logits.max(-1).values - logits.min(-1).values).detach().cpu().tolist()
        )
        h = aux["node_embeddings"]
        batch = obs.batch if obs.batch is not None else torch.zeros(obs.num_nodes, dtype=torch.long)
        z = aux["pooled"]
        z_norms.extend(z.detach().norm(dim=-1).cpu().tolist())
        h_norms.extend(h.detach().norm(dim=-1).cpu().tolist())
        w = _attention_weights(model, h, batch, b)
        sums = torch.zeros(b, device=h.device)
        sums.index_add_(0, batch, w)
        topw = torch.full((b,), float("-inf"), device=h.device)
        topw.scatter_reduce_(0, batch, w, reduce="amax", include_self=False)
        topw = torch.nan_to_num(topw, nan=0.0)
        top1.extend((topw / sums.clamp(min=1e-12)).cpu().tolist())
        # attention share of the two marked nodes per graph
        counts = torch.bincount(batch, minlength=b)
        starts = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1]])
        share_nodes: list[int] = []
        for gi, i in enumerate(idxs):
            base = int(starts[gi])
            for m in dataset.graphs[int(i)].metadata["marked"]:
                share_nodes.append(base + int(m))
        share_acc = torch.zeros(b, device=h.device)
        if share_nodes:
            sel = torch.tensor(share_nodes, device=h.device)
            share_acc.index_add_(0, batch[sel], w[sel])
        marker_share.extend((share_acc / sums.clamp(min=1e-12)).cpu().tolist())
        correct_imm += int((pred_imm == labels).sum().item())

        # --- delayed probe from the same presentation (intact vs reset) ---
        state = model.initial_state(b)
        _, state, _ = model.step(obs, state)
        for _ in range(delay):
            obs_d = observe_instances(dataset, idxs, False, marker_dim, generic_blank)
            _, state, _ = model.step(obs_d, state)
        obs_d = observe_instances(dataset, idxs, False, marker_dim, generic_blank)
        logits_i, _, _ = model.step(obs_d, state)
        state_r = model.reset_state(state) if isinstance(state, torch.Tensor) else state
        logits_r, _, _ = model.step(obs_d, state_r)
        correct_intact += int((logits_i.argmax(-1).cpu() == labels).sum().item())
        correct_reset += int((logits_r.argmax(-1).cpu() == labels).sum().item())
        total += b

    return {
        "immediate_accuracy": correct_imm / max(total, 1),
        "delayed_intact": correct_intact / max(total, 1),
        "delayed_reset": correct_reset / max(total, 1),
        "mean_z_norm": float(np.mean(z_norms)) if z_norms else float("nan"),
        "mean_h_norm": float(np.mean(h_norms)) if h_norms else float("nan"),
        "mean_attn_top1": float(np.mean(top1)) if top1 else float("nan"),
        "mean_marker_attention_share": (
            float(np.mean(marker_share)) if marker_share else float("nan")
        ),
        "mean_logit_margin": float(np.mean(margins)) if margins else float("nan"),
    }


def sweep_size(
    model,
    task_cfg: dict,
    graph_cfg: dict,
    sizes: list[int],
    sets_per_size: int = 2,
    instances_per_set: int = 200,
    delay: int | None = None,
    batch_size: int = 32,
    generic_blank_graph=None,
    base_gen: dict | None = None,
) -> list[dict]:
    """Run the controlled size sweep for one model on a ladder of sizes.

    Returns one row per (size, density_mode, set_idx). ``delay`` defaults to
    the task's trained delay.
    """
    gen = base_gen or _gen_cfg(task_cfg or {}, graph_cfg or {})
    if delay is None:
        delay = int((task_cfg or {}).get("delay", 3))
    base_nodes = int(gen.get("base_nodes", int((task_cfg or {}).get("num_nodes", sizes[0]))))
    base_density = float(gen.get("edge_density"))
    marker_dim = int(gen["marker_dim"])

    rows: list[dict] = []
    for num_nodes in sizes:
        num_nodes = int(num_nodes)
        for mode in ("fixed", "degree_matched"):
            if mode == "fixed":
                density = base_density
            else:
                density = _degree_matched_density(base_density, base_nodes, num_nodes)
            for set_idx in range(sets_per_size):
                seed = 10000 + num_nodes + 977 * set_idx + (7 if mode == "fixed" else 13)
                ds = make_connectivity_dataset(
                    int(instances_per_set),
                    num_nodes=num_nodes,
                    feature_dim=int(gen["feature_dim"]),
                    marker_dim=marker_dim,
                    max_path_len=int(gen["max_path_len"]),
                    edge_density=density,
                    directed=bool(gen["directed"]),
                    seed=seed,
                    label_mode=gen["label_mode"],
                    neg_min_distance=int(gen["neg_min_distance"]),
                )
                res = _probe_size(
                    model, ds, delay=delay, batch_size=batch_size,
                    marker_dim=marker_dim, generic_blank=generic_blank_graph,
                )
                rows.append(
                    {
                        "num_nodes": num_nodes,
                        "density_mode": mode,
                        "edge_density": density,
                        "mean_degree": density * (num_nodes - 1),
                        "set_idx": set_idx,
                        "num_instances": len(ds),
                        **res,
                    }
                )
    return rows
