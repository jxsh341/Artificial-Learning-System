"""H5 — state-to-readout coupling rescue experiment (report §8b).

Three arms share the same starting checkpoints (phenotype-iii seeds:
answer decodable in state, substitution fails, immediate rule intact) and
the same episode budget (200 coupling + 200 restore):

- ``control``      : nothing frozen — persistence of the phenotype under the
  original (D800-style) interleaved training;
- ``treatment_a``  : stage 3 freezes GNN + pooling + the SSM *state-generation*
  pathway (decay ``A_raw``, input projection ``B``) and trains only the
  readout path — SSM output projections ``C``, ``D`` and the head — on
  correctly labelled delayed trials;
- ``treatment_b``  : identical to ``treatment_a`` but the state–target
  association is broken (labels shuffled, same instances): tests whether any
  improvement is coupling-specific rather than extra training.

No new modules, no added state dimensions (user constraint). Success rubric
and outcome taxonomy are pre-registered in TECHNICAL_REPORT §8b.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from artificial_system.data.graph_tasks import ConnectivityDataset, make_connectivity_dataset
from artificial_system.environment.graph_environment import GraphClassificationEnvironment
from artificial_system.evaluation import graph_classification as graph_eval
from artificial_system.evaluation import state_content as state_eval


def set_trainables(model, mode: str) -> None:
    """Freeze schedule for the coupling stage.

    ``readout_only``: trainable = SSM output projections (C, D) + head.
    Frozen = graph_encoder, pooling gate, SSM state generation (A_raw/diagonal
    decay + input projection B) — i.e. the already-formed answer producing
    pathway is untouched.
    ``all``: everything trainable (restore stage / control arm).
    """
    train_all = mode == "all"
    for p in model.graph_encoder.parameters():
        p.requires_grad_(train_all)
    for p in model.pooling.parameters():
        p.requires_grad_(train_all)
    ssm = model.temporal_model
    for name, p in ssm.named_parameters():
        is_readout = name.startswith("C.") or name.startswith("D.")
        p.requires_grad_(train_all or is_readout)
    for p in model.head.parameters():
        p.requires_grad_(train_all or mode == "readout_only")


def relabel_dataset(ds: ConnectivityDataset, seed: int) -> ConnectivityDataset:
    """Treatment B: shuffle labels (same graphs/features/states, broken
    state–target association)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ds))
    return ConnectivityDataset(
        graphs=list(ds.graphs),
        labels=ds.labels[torch.as_tensor(perm, dtype=torch.long)].clone(),
        metadata=dict(ds.metadata),
    )


def build_datasets_from_cfg(cfg: dict) -> dict:
    task_cfg = dict(cfg.get("task", {}))
    gen_cfg = dict(
        num_nodes=int(task_cfg.get("num_nodes", 100)),
        feature_dim=int(cfg.get("graph", {}).get("node_feature_dim", 2)),
        marker_dim=int(task_cfg.get("marker_dim", 2)),
        max_path_len=int(task_cfg.get("max_path_len", 3)),
        edge_density=float(task_cfg.get("edge_density", 0.04)),
        directed=bool(task_cfg.get("directed", False)),
        label_mode=task_cfg.get("label_mode", "adjacency"),
        neg_min_distance=int(task_cfg.get("neg_min_distance", 3)),
    )
    ds_seed = int(task_cfg.get("dataset_seed", 123))
    return {
        "train": make_connectivity_dataset(int(task_cfg.get("num_train", 2000)), seed=ds_seed, **gen_cfg),
        "test": make_connectivity_dataset(int(task_cfg.get("num_test", 200)), seed=ds_seed + 1, **gen_cfg),
    }


@torch.no_grad()
def quick_probe(model, test_ds, delay: int, marker_dim: int, generic_blank) -> dict:
    imm = graph_eval.probe_dataset(
        model, test_ds, delay=0, batch_size=32, marker_dim=marker_dim
    )["accuracy"]
    del_probe = graph_eval.probe_delayed_intact_vs_reset(
        model, test_ds, delay=delay, batch_size=32,
        marker_dim=marker_dim, generic_blank=generic_blank,
    )
    return {
        "immediate": imm,
        "delayed_intact": del_probe["delayed_intact"],
        "delayed_reset": del_probe["delayed_state_reset"],
    }


def _final_battery(model, test_ds, delay, marker_dim, generic_blank, seed) -> dict:
    battery: dict = {}
    battery.update(quick_probe(model, test_ds, delay, marker_dim, generic_blank))
    rows, cf = graph_eval.run_counterfactual_probe(
        model, test_ds, delay=delay, batch_size=32,
        marker_dim=marker_dim, generic_blank=generic_blank, seed=seed,
    )
    battery.update({f"cf_{k}": v for k, v in cf.items()})
    _, locus = graph_eval.run_graph_locus(
        model, test_ds, delay=delay, batch_size=32,
        marker_dim=marker_dim, generic_blank=generic_blank,
    )
    battery.update(locus)
    _, rw = graph_eval.run_rewired_control(
        model, test_ds, levels=[1.0], batch_size=32, seed=seed,
    )
    battery.update(rw)
    return battery


def run_coupling_arm(
    cfg: dict,
    datasets: dict,
    ckpt: dict,
    arm: str,
    run_dir: Path,
    seed: int,
    schedule_spec: dict,
) -> dict:
    """One arm on one starting checkpoint. Record with logger-style CSV/JSON
    in ``run_dir``; returns the summary."""
    from artificial_system.models.gnn_ssm import build_model
    from artificial_system.training.trainer import Trainer

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    task_cfg = cfg.get("task", {})
    train_cfg = dict(cfg.get("training", {}))
    delay = int(task_cfg.get("delay", 3))
    marker_dim = int(task_cfg.get("marker_dim", 2))

    model = build_model(
        cfg, node_feature_dim=int(cfg.get("graph", {}).get("node_feature_dim", 2)),
        edge_feature_dim=0, num_actions=2, num_nodes=None,
    )
    model.load_state_dict(ckpt["model_state"])

    test_ds = datasets["test"]
    generic_blank = None
    if task_cfg.get("blank_mode", "no_markers") == "generic":
        from artificial_system.environment.graph_environment import make_generic_graph
        generic_blank = make_generic_graph(
            int(task_cfg.get("num_nodes", 100)),
            int(cfg.get("graph", {}).get("node_feature_dim", 2)),
            seed * 2000 + 2,
        )

    env_del = GraphClassificationEnvironment(datasets["train"], task_cfg, seed=seed * 2000 + 2)
    imm_cfg = dict(task_cfg)
    imm_cfg["delay"] = 0
    env_imm = GraphClassificationEnvironment(datasets["train"], imm_cfg, seed=seed * 2000 + 1)
    # treatment B trains the readout on broken associations (same instances)
    env_shuf = None
    if arm == "treatment_b":
        shuffled = relabel_dataset(datasets["train"], seed + 777)
        env_shuf = GraphClassificationEnvironment(shuffled, task_cfg, seed=seed * 2000 + 2)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    tr_del = Trainer(model, env_del, optimizer, train_cfg)
    tr_imm = Trainer(model, env_imm, optimizer, train_cfg)
    tr_shuf = Trainer(model, env_shuf, optimizer, train_cfg) if env_shuf is not None else None

    # ---------------- pre-intervention analysis ----------------
    _, pre_summary = state_eval.run_state_content(
        model, test_ds, delay=delay, batch_size=50, marker_dim=marker_dim,
        generic_blank=generic_blank, seed=seed,
    )
    pre_summary = {k: v for k, v in pre_summary.items()
                   if isinstance(v, (int, float)) and np.isfinite(v)}

    chunk = max(1, int(schedule_spec.get("probe_every", 25)))
    rows: list[dict] = []

    def probe(tag: str, ep: int) -> None:
        quick = quick_probe(model, test_ds, delay, marker_dim, generic_blank)
        rows.append({"phase": tag, "episodes_done": ep, **quick})

    # ---------------- stage 3: coupling ----------------
    stage3 = schedule_spec.get("stage3", {})
    stage4 = schedule_spec.get("stage4", {})
    # arm-specific coupling trainer
    if arm == "control":
        set_trainables(model, "all")
        coupling_trainers = [("interleaved", tr_imm, tr_del)]
    else:
        set_trainables(model, "readout_only")
        if arm == "treatment_b":
            coupling_trainers = [("delayed", None, tr_shuf)]
        else:
            coupling_trainers = [("delayed", None, tr_del)]

    ep_total = 0
    for phase, tr_imm_arm, tr_del_arm in coupling_trainers:
        n = int(stage3.get("episodes", 200))
        if tr_imm_arm is None:
            tr_del_arm.train(n)
            ep_total += n
        else:
            tr_imm_arm.train(n // 2)
            tr_del_arm.train(n - n // 2)
            ep_total += n

    probe("post_coupling", 0)

    # ---------------- stage 4: restore normal interleaved ----------------
    set_trainables(model, "all")
    restore_eps = int(stage4.get("episodes", 200))
    tr_imm.train(restore_eps // 2)
    tr_del.train(restore_eps - restore_eps // 2)
    ep_total += restore_eps

    # ---------------- post analysis ----------------
    _, post_summary = state_eval.run_state_content(
        model, test_ds, delay=delay, batch_size=50,
        marker_dim=marker_dim, generic_blank=generic_blank, seed=seed,
    )
    post_summary = {k: v for k, v in post_summary.items()
                    if isinstance(v, (int, float)) and np.isfinite(v)}
    battery = _final_battery(model, test_ds, delay, marker_dim, generic_blank, seed)

    # success rubric (pre-registered §8b)
    margin_pre = float(pre_summary.get("avf_mean_pre_response_norm", np.nan))
    margin_post = float(post_summary.get("avf_mean_pre_response_norm", np.nan))
    dec_post = post_summary.get("decode_pre_response", np.nan)
    rubric = {
        "immediate_ok": bool(battery["immediate"] >= 0.90),
        "delayed_ok": bool(battery["delayed_intact"] >= 0.90),
        "subst_same_ok": bool(post_summary.get("subst_same_answer_accuracy", 0.0) >= 0.90),
        "subst_diff_follows": float(post_summary.get("subst_diff_follows", 0.0)),
        "zero_chance": bool(post_summary.get("subst_zero_accuracy", 1.0) <= 0.65),
        "shuffled_chance": bool(post_summary.get("subst_shuffled_accuracy", 1.0) <= 0.65),
        "decode_preserved": bool(dec_post >= 0.95),
        "state_norm_ratio_post_over_pre": margin_post / margin_pre if margin_pre else np.nan,
    }
    rubric["all_criteria"] = all(
        rubric[k] for k in
        ("immediate_ok", "delayed_ok", "subst_same_ok", "zero_chance", "shuffled_chance")
    ) and rubric["subst_diff_follows"] >= 0.9 and rubric["decode_preserved"]

    summary = {
        "arm": arm,
        "seed": seed,
        "source_checkpoint": ckpt.get("source_path"),
        "recorded_delayed_pre": ckpt.get("recorded_delayed"),
        "pre_analysis": pre_summary,
        "post_analysis": post_summary,
        "battery": battery,
        "rubric": rubric,
        "episodes_coupling": int(stage3.get("episodes", 200)),
        "episodes_restore": restore_eps,
    }
    pd.DataFrame(rows).to_csv(run_dir / "training_probes.csv", index=False)
    import json

    def _py(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(repr(o))

    (run_dir / "arm_summary.json").write_text(json.dumps(summary, indent=2, default=_py))
    return summary
