"""Graph-dependent classification evaluations.

- :func:`probe_dataset`: task-consistent classification probe (overall,
  per class, per path-length bucket) — the failure-mode view (spec §21 §6).
- :func:`run_rewired_control`: the M7 diagnostic. Rewire test edges and
  probe twice: with *recomputed* labels (should stay high if the model
  reads connectivity from the presented graph) and with the *original*
  labels (should collapse to chance if predictions follow structure
  rather than memorised instances).
- :func:`run_graph_locus`: state-retention vs recomputation dissociation
  for the delayed relational task.
- :func:`run_graph_generalization`: unseen instances and unseen graph size.
"""
from __future__ import annotations

import torch
from torch import Tensor

from artificial_system.data.graph_tasks import recompute_label
from artificial_system.data.schema import GraphData

IGNORE_INDEX = -100


def observe_instances(
    dataset,
    idxs,
    keep_markers: bool = True,
    marker_dim: int = 2,
    generic_blank: GraphData | None = None,
) -> GraphData:
    """Build a batched observation from dataset instances (env-free).

    When ``generic_blank`` is provided and ``keep_markers`` is False, the
    shared instance-independent blank graph is used instead (matching a
    ``blank_mode: generic`` training regime).
    """
    from artificial_system.data.loaders import collate_graphs

    if not keep_markers and generic_blank is not None:
        return collate_graphs([generic_blank] * len(list(idxs)))
    graphs = []
    for i in idxs:
        g = dataset.graphs[int(i)]
        if keep_markers:
            graphs.append(g)
        else:
            feats = g.node_features.clone()
            feats[:, :marker_dim] = 0.0
            graphs.append(GraphData(node_features=feats, edge_index=g.edge_index, metadata=dict(g.metadata)))
    return collate_graphs(graphs)


def _chunks(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        yield list(range(start, min(start + batch_size, n)))


@torch.no_grad()
def probe_dataset(
    model,
    dataset,
    delay: int = 0,
    batch_size: int = 16,
    marker_dim: int | None = None,
    generic_blank: GraphData | None = None,
) -> dict:
    """Task-consistent probe: presentation -> (delay blanks) -> response.

    Returns overall accuracy, per-class accuracy, and accuracy bucketed by
    the true path length of positive instances.
    """
    if marker_dim is None:
        marker_dim = int(dataset.metadata.get("config", {}).get("marker_dim", 2))
    model.eval()
    preds = []
    for idxs in _chunks(len(dataset), batch_size):
        obs = observe_instances(dataset, idxs, keep_markers=True, marker_dim=marker_dim)
        if delay == 0:
            logits, _, _ = model.step(obs, None)
        else:
            state = model.initial_state(len(idxs))
            _, state, _ = model.step(obs, state)
            for _ in range(delay):
                obs = observe_instances(dataset, idxs, keep_markers=False, marker_dim=marker_dim, generic_blank=generic_blank)
                _, state, _ = model.step(obs, state)
            obs = observe_instances(dataset, idxs, keep_markers=False, marker_dim=marker_dim, generic_blank=generic_blank)
            logits, _, _ = model.step(obs, state)
        preds.append(logits.argmax(-1).cpu())
    preds = torch.cat(preds)
    labels = dataset.labels
    acc = float((preds == labels).float().mean().item())

    per_path_len: dict[str, float] = {}
    instances = dataset.metadata.get("instances", [])
    for d in sorted({i["path_len"] for i in instances if i["path_len"] is not None}):
        sel = [k for k, i in enumerate(instances) if i["path_len"] == d]
        hit = (preds[sel] == labels[sel]).float().mean().item()
        per_path_len[str(d)] = float(hit)
    per_class = {}
    for c in (0, 1):
        sel = (labels == c).nonzero(as_tuple=True)[0]
        per_class[str(c)] = float((preds[sel] == labels[sel]).float().mean().item())
    model.train()
    return {
        "accuracy": acc,
        "accuracy_by_path_len": per_path_len,
        "accuracy_by_class": per_class,
    }


@torch.no_grad()
def run_rewired_control(
    model,
    dataset,
    levels: list[float] | None = None,
    batch_size: int = 16,
    seed: int = 0,
) -> tuple[list[dict], dict]:
    """M7 diagnostic: does the model's prediction depend on the correct edges?

    For each rewiring level:
      (a) ``relabeled``: rewire edges AND recompute labels from the rewired
          graph — accuracy should stay high if the model reads connectivity;
      (b) ``original_labels``: rewire edges, keep original labels — accuracy
          should collapse to chance if predictions follow structure.
    """
    from artificial_system.data.perturbations import randomize_edges, swap_edges
    from artificial_system.data.loaders import collate_graphs

    levels = list(levels or [0.25, 0.5, 1.0])
    marker_dim = int(dataset.metadata.get("config", {}).get("marker_dim", 2))
    max_path_len = int(dataset.metadata.get("config", {}).get("max_path_len", 3))
    label_mode = dataset.metadata.get("config", {}).get("label_mode", "path")

    rows: list[dict] = []
    intact = probe_dataset(model, dataset, batch_size=batch_size, marker_dim=marker_dim)
    rows.append({"condition": "intact", "level": 0.0, "relabeled": True, "accuracy": intact["accuracy"]})

    for perturb_name, perturb_fn in (("rewired", randomize_edges), ("swapped", swap_edges)):
        for level in levels:
            # (a) perturb + recompute labels
            graphs2, labels2 = [], []
            for g in dataset.graphs:
                g2 = perturb_fn(g, level, seed)
                labels2.append(recompute_label(g2, g.metadata["marked"], max_path_len, label_mode))
                graphs2.append(g2)

            from artificial_system.data.graph_tasks import ConnectivityDataset

            relabeled_ds = ConnectivityDataset(
                graphs2, torch.tensor(labels2, dtype=torch.long), metadata=dataset.metadata
            )
            acc_relabeled = probe_dataset(model, relabeled_ds, batch_size=batch_size, marker_dim=marker_dim)["accuracy"]
            # (b) perturb, original labels
            original_ds = ConnectivityDataset(
                graphs2, dataset.labels.clone(), metadata=dataset.metadata
            )
            acc_original = probe_dataset(model, original_ds, batch_size=batch_size, marker_dim=marker_dim)["accuracy"]
            rows.append({"condition": f"{perturb_name}_relabeled", "level": level, "relabeled": True, "accuracy": acc_relabeled})
            rows.append({"condition": f"{perturb_name}_original_labels", "level": level, "relabeled": False, "accuracy": acc_original})

    summary: dict = {"rewired_intact_accuracy": intact["accuracy"]}
    for row in rows[1:]:
        summary[f"rewired_{row['condition']}@{row['level']}_accuracy"] = row["accuracy"]
    return rows, summary


@torch.no_grad()
def run_graph_locus(
    model,
    dataset,
    delay: int = 2,
    batch_size: int = 16,
    marker_dim: int | None = None,
    generic_blank: GraphData | None = None,
) -> tuple[list[dict], dict]:
    """State-retention vs recomputation dissociation (delayed relational task).

    Conditions:
      - ``locus_immediate``: response at presentation (no memory needed);
      - ``locus_delayed_retain``: markers cleared during delay, state intact
        — requires state retention of the computed answer;
      - ``locus_delayed_reset_cleared``: same, but state reset before the
        response — nothing left to answer with (chance expected);
      - ``locus_delayed_markers_kept``: markers visible throughout — retain
        or recompute;
      - ``locus_delayed_markers_kept_reset``: markers visible, state reset
        — must recompute from the presented graph at response time.

    The contrast between the last two dissociates state retention from
    recomputation-from-persistent-input.
    """
    if marker_dim is None:
        marker_dim = int(dataset.metadata.get("config", {}).get("marker_dim", 2))
    model.eval()
    conditions = ["immediate", "delayed_retain", "delayed_reset_cleared", "delayed_markers_kept", "delayed_markers_kept_reset"]
    counts = {c: [0, 0] for c in conditions}

    for idxs in _chunks(len(dataset), batch_size):
        labels = dataset.labels[torch.as_tensor(idxs, dtype=torch.long)]
        b = len(idxs)

        # immediate: response at presentation
        obs = observe_instances(dataset, idxs, True, marker_dim)
        logits, _, _ = model.step(obs, None)
        pred = logits.argmax(-1).cpu()
        counts["immediate"][0] += int((pred == labels).sum().item())
        counts["immediate"][1] += b

        def delayed(state_reset: bool, keep_markers: bool) -> tuple[int, int]:
            obs = observe_instances(dataset, idxs, True, marker_dim)
            state = model.initial_state(b)
            _, state, _ = model.step(obs, state)
            for _ in range(delay):
                obs = observe_instances(dataset, idxs, keep_markers, marker_dim, generic_blank)
                if state_reset and isinstance(state, Tensor):
                    state = model.reset_state(state)
                _, state, _ = model.step(obs, state)
            if state_reset and isinstance(state, Tensor):
                state = model.reset_state(state)
            obs = observe_instances(dataset, idxs, keep_markers, marker_dim, generic_blank)
            logits, _, _ = model.step(obs, state)
            pred = logits.argmax(-1).cpu()
            return int((pred == labels).sum().item()), b

        c, t = delayed(False, False)
        counts["delayed_retain"][0] += c
        counts["delayed_retain"][1] += t
        c, t = delayed(True, False)
        counts["delayed_reset_cleared"][0] += c
        counts["delayed_reset_cleared"][1] += t
        c, t = delayed(False, True)
        counts["delayed_markers_kept"][0] += c
        counts["delayed_markers_kept"][1] += t
        c, t = delayed(True, True)
        counts["delayed_markers_kept_reset"][0] += c
        counts["delayed_markers_kept_reset"][1] += t

    summary = {f"locus_{c}_accuracy": counts[c][0] / max(counts[c][1], 1) for c in conditions}
    rows = [{"condition": c, "accuracy": summary[f"locus_{c}_accuracy"]} for c in conditions]
    model.train()
    return rows, summary


@torch.no_grad()
def run_graph_generalization(
    model,
    test_dataset,
    unseen_size_dataset=None,
    delay: int = 0,
    batch_size: int = 16,
) -> tuple[list[dict], dict]:
    """Probe on unseen instances (test split) and unseen graph size."""
    marker_dim = int(test_dataset.metadata.get("config", {}).get("marker_dim", 2))
    rows: list[dict] = []
    summary: dict = {}
    acc = probe_dataset(model, test_dataset, delay=delay, batch_size=batch_size, marker_dim=marker_dim)
    rows.append({"split": "test_unseen_instances", "num_nodes": test_dataset.metadata["config"]["num_nodes"], "accuracy": acc["accuracy"]})
    summary["graph_generalization_unseen_instances"] = acc["accuracy"]
    if unseen_size_dataset is not None:
        acc2 = probe_dataset(model, unseen_size_dataset, delay=delay, batch_size=batch_size)
        rows.append({"split": "test_unseen_size", "num_nodes": unseen_size_dataset.metadata["config"]["num_nodes"], "accuracy": acc2["accuracy"]})
        summary["graph_generalization_unseen_size"] = acc2["accuracy"]
        summary["graph_generalization_size_gap"] = acc["accuracy"] - acc2["accuracy"]
    return rows, summary


@torch.no_grad()
def probe_delayed_intact_vs_reset(
    model,
    dataset,
    delay: int = 3,
    batch_size: int = 16,
    marker_dim: int | None = None,
    generic_blank: GraphData | None = None,
) -> dict:
    """Delayed-task accuracy with intact state vs state reset before the
    response, from the same forward pass (used throughout training — the
    intact-vs-reset difference traces when the state starts carrying the
    computed answer)."""
    if marker_dim is None:
        marker_dim = int(dataset.metadata.get("config", {}).get("marker_dim", 2))
    model.eval()
    intact_c = reset_c = total = 0
    for idxs in _chunks(len(dataset), batch_size):
        labels = dataset.labels[torch.as_tensor(idxs, dtype=torch.long)]
        b = len(idxs)
        obs = observe_instances(dataset, idxs, True, marker_dim)
        state = model.initial_state(b)
        _, state, _ = model.step(obs, state)
        for _ in range(delay):
            obs = observe_instances(dataset, idxs, False, marker_dim, generic_blank)
            _, state, _ = model.step(obs, state)
        obs = observe_instances(dataset, idxs, False, marker_dim, generic_blank)
        logits_i, _, _ = model.step(obs, state)
        s_reset = model.reset_state(state) if isinstance(state, Tensor) else state
        logits_r, _, _ = model.step(obs, s_reset)
        pred_i = logits_i.argmax(-1).cpu()
        pred_r = logits_r.argmax(-1).cpu()
        intact_c += int((pred_i == labels).sum().item())
        reset_c += int((pred_r == labels).sum().item())
        total += b
    model.train()
    return {
        "delayed_intact": intact_c / max(total, 1),
        "delayed_state_reset": reset_c / max(total, 1),
    }


@torch.no_grad()
def run_delay_retention(
    model,
    dataset,
    delays: list[int] | None = None,
    batch_size: int = 16,
    marker_dim: int | None = None,
    generic_blank: GraphData | None = None,
) -> tuple[list[dict], dict]:
    """Memory curve of the COMPUTED answer: presentation -> d blanks ->
    response, accuracy as a function of temporal distance."""
    delays = list(delays or [0, 3, 10, 30, 100])
    if marker_dim is None:
        marker_dim = int(dataset.metadata.get("config", {}).get("marker_dim", 2))
    model.eval()
    rows: list[dict] = []
    for d in delays:
        d = int(d)
        correct = total = 0
        for idxs in _chunks(len(dataset), batch_size):
            labels = dataset.labels[torch.as_tensor(idxs, dtype=torch.long)]
            b = len(idxs)
            obs = observe_instances(dataset, idxs, True, marker_dim)
            state = model.initial_state(b)
            _, state, _ = model.step(obs, state)
            for _ in range(d):
                obs = observe_instances(dataset, idxs, False, marker_dim, generic_blank)
                _, state, _ = model.step(obs, state)
            obs = observe_instances(dataset, idxs, False, marker_dim, generic_blank)
            logits, _, _ = model.step(obs, state)
            pred = logits.argmax(-1).cpu()
            correct += int((pred == labels).sum().item())
            total += b
        rows.append({"delay": d, "accuracy": correct / max(total, 1)})
    summary = {f"answer_retention@{row['delay']}": row["accuracy"] for row in rows}
    model.train()
    return rows, summary


@torch.no_grad()
def run_counterfactual_probe(
    model,
    dataset,
    delay: int = 3,
    batch_size: int = 16,
    marker_dim: int | None = None,
    generic_blank: GraphData | None = None,
    seed: int = 0,
) -> tuple[list[dict], dict]:
    """Delayed counterfactual probe: rigid retention vs flexible updating.

    Trial: presentation (instance i, markers visible) -> delay (markers
    cleared) -> revision cue -> response.

      - ``no_change``:  revision = same instance re-presented
        -> correct answer = label(i) (retain or recompute; both agree);
      - ``update``:     revision = a different instance j
        -> correct answer = label(j); on label-flipped pairs the answer
        MUST change — tests whether the model updates on new evidence;
      - ``irrelevant``: revision = instance-independent blank
        -> correct answer = label(i) — tests whether irrelevant new input
        destroys the retained answer (confabulation check).
    """
    import numpy as np

    if marker_dim is None:
        marker_dim = int(dataset.metadata.get("config", {}).get("marker_dim", 2))
    rng = np.random.default_rng(seed)
    model.eval()
    conditions = ["no_change", "update", "irrelevant"]
    counts = {c: [0, 0] for c in conditions}
    flipped = [0, 0]  # update condition restricted to label(i) != label(j)
    same_label = [0, 0]

    n = len(dataset)
    for idxs in _chunks(n, batch_size):
        b = len(idxs)
        idxs_j = [int(rng.choice([k for k in range(n) if k != i])) for i in idxs]
        labels_i = dataset.labels[torch.as_tensor(idxs, dtype=torch.long)]
        labels_j = dataset.labels[torch.as_tensor(idxs_j, dtype=torch.long)]

        obs_pres = observe_instances(dataset, idxs, True, marker_dim)
        state = model.initial_state(b)
        _, state, _ = model.step(obs_pres, state)
        for _ in range(delay):
            obs = observe_instances(dataset, idxs, False, marker_dim, generic_blank)
            _, state, _ = model.step(obs, state)

        # one response per revision condition, from the same post-delay state
        revisions = {
            "no_change": (observe_instances(dataset, idxs, True, marker_dim), labels_i),
            "update": (observe_instances(dataset, idxs_j, True, marker_dim), labels_j),
            "irrelevant": (observe_instances(dataset, idxs, False, marker_dim, generic_blank), labels_i),
        }
        for cond, (obs_rev, target) in revisions.items():
            logits, _, _ = model.step(obs_rev, state)
            pred = logits.argmax(-1).cpu()
            hit = pred == target
            counts[cond][0] += int(hit.sum().item())
            counts[cond][1] += b
            if cond == "update":
                flip_mask = labels_i != labels_j
                if bool(flip_mask.any()):
                    flipped[0] += int(hit[flip_mask].sum().item())
                    flipped[1] += int(flip_mask.sum().item())
                same_mask = ~flip_mask
                if bool(same_mask.any()):
                    same_label[0] += int(hit[same_mask].sum().item())
                    same_label[1] += int(same_mask.sum().item())
    model.train()
    summary = {f"counterfactual_{c}_accuracy": counts[c][0] / max(counts[c][1], 1) for c in conditions}
    summary["counterfactual_update_flipped_accuracy"] = flipped[0] / max(flipped[1], 1)
    summary["counterfactual_update_samelabel_accuracy"] = same_label[0] / max(same_label[1], 1)
    rows = [
        {"condition": c, "accuracy": summary[f"counterfactual_{c}_accuracy"]}
        for c in conditions
    ] + [
        {"condition": "update_label_flipped", "accuracy": summary["counterfactual_update_flipped_accuracy"]},
        {"condition": "update_label_same", "accuracy": summary["counterfactual_update_samelabel_accuracy"]},
    ]
    return rows, summary


