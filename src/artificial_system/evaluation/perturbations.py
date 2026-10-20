"""Perturbation experiments (spec §19, phase 3-4: "attack the system").

Temporal attacks use the SSM reset machinery:

    normal -> full state reset -> partial state reset -> noise injection
        -> history truncation

The :func:`storage_locus` analysis crosses temporal state (intact/reset)
with graph representation (intact/features-shuffled) to determine where
learned information resides:

- weights only,
- instantaneous graph representation,
- SSM state,
- a distributed combination.
"""
from __future__ import annotations

import torch

from artificial_system.data.perturbations import (
    drop_community,
    randomize_edges,
    remove_edges,
    remove_nodes,
    perturb_weights,
    shuffle_features,
)


@torch.no_grad()
def probe_task(
    model,
    env,
    num_steps: int,
    state_transform=None,
    batch_size: int = 16,
) -> dict:
    """Accuracy over task trials with an optional per-step state transformation.

    ``state_transform(state, step) -> state`` is applied before every model
    step (e.g. state resets, noise injection).
    """
    model.eval()
    obs = env.reset(batch_size)
    state = model.initial_state(batch_size)
    correct = total = 0
    norms: list[float] = []
    for step in range(num_steps):
        if state_transform is not None:
            state = state_transform(state, step)
        logits, state, _ = model.step(obs, state)
        targets = env.current_targets()
        active = targets != -100
        if bool(active.any()):
            pred = logits.argmax(-1)
            correct += int((pred[active] == targets[active]).sum().item())
            total += int(active.sum().item())
        obs, _, _, _ = env.step(logits.argmax(-1))
        if isinstance(state, torch.Tensor):
            norms.append(float(state.norm(dim=-1).mean().item()))
    model.train()
    return {
        "accuracy": correct / max(total, 1),
        "state_norm_mean": float(sum(norms) / len(norms)) if norms else float("nan"),
    }


# ------------------------------------------------------------ state transforms


def _state_reset(model):
    def fn(state, step):
        if isinstance(state, torch.Tensor):
            return model.reset_state(state)
        return state

    return fn


def _partial_reset(fraction: float):
    def fn(state, step):
        if not isinstance(state, torch.Tensor) or fraction <= 0:
            return state
        dim = state.shape[-1]
        k = int(round(fraction * dim))
        if k == 0:
            return state
        idx = torch.randperm(dim, device=state.device)[:k]
        state = state.clone()
        state[:, idx] = 0.0
        return state

    return fn


def _noise(sigma: float):
    def fn(state, step):
        if isinstance(state, torch.Tensor) and sigma > 0:
            return state + sigma * torch.randn_like(state)
        return state

    return fn


def _truncation(model, interval: int):
    def fn(state, step):
        if isinstance(state, torch.Tensor) and interval > 0 and step > 0 and step % interval == 0:
            return model.reset_state(state)
        return state

    return fn


# ------------------------------------------------------------ temporal attacks


def run_temporal_perturbations(
    model,
    env,
    num_steps: int = 200,
    batch_size: int = 16,
    noise_sigmas: list[float] | None = None,
    reset_fractions: list[float] | None = None,
    truncation_intervals: list[int] | None = None,
) -> tuple[list[dict], dict]:
    """Normal -> full reset -> partial resets -> noise -> truncation."""
    noise_sigmas = list(noise_sigmas or [0.1, 0.5, 1.0])
    reset_fractions = list(reset_fractions or [0.25, 0.5, 0.75])
    truncation_intervals = list(truncation_intervals or [2, 5, 10])

    rows: list[dict] = []
    intact = probe_task(model, env, num_steps, batch_size=batch_size)
    rows.append({"perturbation": "none", "level": 0.0, **intact})

    probes: list[tuple[str, float, dict]] = [
        (
            "state_reset",
            1.0,
            probe_task(model, env, num_steps, state_transform=_state_reset(model), batch_size=batch_size),
        )
    ]
    for f in reset_fractions:
        probes.append(
            ("partial_reset", float(f),
             probe_task(model, env, num_steps, state_transform=_partial_reset(f), batch_size=batch_size))
        )
    for sigma in noise_sigmas:
        probes.append(
            ("noise", float(sigma),
             probe_task(model, env, num_steps, state_transform=_noise(sigma), batch_size=batch_size))
        )
    for k in truncation_intervals:
        probes.append(
            ("truncate", float(k),
             probe_task(model, env, num_steps, state_transform=_truncation(model, k), batch_size=batch_size))
        )
    rows.extend({"perturbation": name, "level": level, **rec} for name, level, rec in probes)

    summary = {f"temporal_{name}@{level}_accuracy": rec["accuracy"] for name, level, rec in probes}
    summary["temporal_intact_accuracy"] = intact["accuracy"]
    return rows, summary


# ------------------------------------------------------------ storage locus


def storage_locus(
    model,
    build_env,
    base_graph,
    num_steps: int = 200,
    batch_size: int = 16,
    seed: int = 0,
    delay_variants: tuple[int | None, ...] = (None, 0),
) -> tuple[list[dict], dict]:
    """Where does learned information reside? (phase 3)

    Crosses temporal state (intact / reset) with graph representation
    (intact / features-shuffled), under two response regimes:

    - the *trained delay* (``None``): the state's role in bridging the
      cue-to-response gap is visible;
    - *immediate* (``delay=0``): the learned mapping must reside in weights
      (+ cue-in-input) if performance survives state resets.

    ``build_env(graph, seed, task_overrides) -> env`` must construct an
    environment bound to the given graph (the runner provides this).
    """
    keys = {
        (None, "intact", "state_intact"): "locus_full",
        (None, "intact", "state_reset"): "locus_no_state",
        (None, "shuffled", "state_intact"): "locus_no_graph",
        (None, "shuffled", "state_reset"): "locus_weights_only",
        (0, "intact", "state_intact"): "locus_immediate_full",
        (0, "intact", "state_reset"): "locus_immediate_no_state",
        (0, "shuffled", "state_intact"): "locus_immediate_no_graph",
        (0, "shuffled", "state_reset"): "locus_immediate_weights_only",
    }
    rows: list[dict] = []
    summary: dict = {}
    for delay in delay_variants:
        for graph_label, graph in (
            ("intact", base_graph),
            ("shuffled", shuffle_features(base_graph, seed=seed)),
        ):
            for state_label, reset in (("state_intact", False), ("state_reset", True)):
                env = build_env(graph, seed, {"delay": delay} if delay is not None else None)
                rec = probe_task(
                    model,
                    env,
                    num_steps,
                    state_transform=_state_reset(model) if reset else None,
                    batch_size=batch_size,
                )
                key = keys[(delay, graph_label, state_label)]
                rows.append({
                    "delay_variant": "immediate" if delay == 0 else "trained_delay",
                    "graph": graph_label,
                    "state": state_label,
                    "locus": key,
                    **rec,
                })
                summary[key] = rec["accuracy"]
                summary[f"{key}_state_norm"] = rec["state_norm_mean"]
    return rows, summary


# ------------------------------------------------------------ graph attacks


def run_graph_perturbations(
    model,
    build_env,
    base_graph,
    probe_steps: int = 200,
    batch_size: int = 16,
    seed: int = 0,
    edge_levels: list[float] | None = None,
    node_levels: list[float] | None = None,
    weight_levels: list[float] | None = None,
) -> tuple[list[dict], dict]:
    """Attack the graph and measure behavioural degradation.

    ``build_env(graph, seed) -> env`` as in :func:`storage_locus`.
    """
    edge_levels = list(edge_levels or [0.25, 0.5, 0.75])
    node_levels = list(node_levels or [0.1, 0.25, 0.5])
    weight_levels = list(weight_levels or [0.25, 0.5, 1.0])

    env0 = build_env(base_graph, seed)
    intact = probe_task(model, env0, probe_steps, batch_size=batch_size)["accuracy"]

    specs = [
        ("remove_edges", remove_edges, edge_levels),
        ("randomize_edges", randomize_edges, edge_levels),
        ("perturb_weights", perturb_weights, weight_levels),
        ("shuffle_features", shuffle_features, [1.0]),
        ("remove_nodes", remove_nodes, node_levels),
        ("drop_community", drop_community, [1.0]),
    ]
    rows = [{"perturbation": "none", "level": 0.0, "accuracy": intact, "delta": 0.0}]
    for name, fn, levels in specs:
        for level in levels:
            graph2 = fn(base_graph, level, seed)
            env2 = build_env(graph2, seed)
            acc = probe_task(model, env2, probe_steps, batch_size=batch_size)["accuracy"]
            rows.append({"perturbation": name, "level": level, "accuracy": acc, "delta": acc - intact})

    summary: dict = {"graph_intact_accuracy": intact}
    for row in rows[1:]:
        summary[f"graph_{row['perturbation']}@{row['level']}_accuracy"] = row["accuracy"]
        summary[f"graph_{row['perturbation']}@{row['level']}_delta"] = row["delta"]
    return rows, summary
