"""H5/H6 — state-to-readout coupling experiments (report §8b, §8c).

Three H5 arms share the same starting checkpoints (phenotype-iii seeds:
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

H6 arms (§8c, capability retention under coupling + restoration):

- ``h6_control``       : H5 treatment A re-run under extended instrumentation;
- ``h6_rehearsal``     : coupling stage single immediate episode every K
  delayed episodes (``rehearsal_every``) — co-adaptation from the start;
- ``h6_freeze_readout``: coupling as H5-A; restore stage freezes head + SSM
  output (C, D) and adapts only the representation under joint episodes.

No new modules, no added state dimensions (user constraint). Success rubrics
and outcome taxonomies are pre-registered in TECHNICAL_REPORT §8b/§8c.
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


READOUT_SSM_PREFIXES = ("C.", "D.")


def set_trainables(model, mode: str) -> None:
    """Freeze schedule.

    ``readout_only``: trainable = SSM output projections (C, D) + head.
      Frozen = graph_encoder, pooling gate, SSM state generation (decay +
      input projection B) — the already-formed answer-producing pathway.
    ``encoder_only``: trainable = everything EXCEPT head and SSM output
      (C, D) — used by the H6 ``h6_freeze_readout`` restore stage (joint
      adaptation under a static readout).
    ``all``: everything trainable (restore stage / control arm).
    """
    if mode == "readout_only":
        rev = False
    elif mode == "encoder_only":
        rev = True
    else:
        rev = None
    train_all = mode == "all"
    for p in model.graph_encoder.parameters():
        p.requires_grad_(train_all if rev is None else rev)
    for p in model.pooling.parameters():
        p.requires_grad_(train_all if rev is None else rev)
    ssm = model.temporal_model
    for name, p in ssm.named_parameters():
        is_readout = name.startswith(READOUT_SSM_PREFIXES)
        if mode == "readout_only":
            p.requires_grad_(is_readout)
        elif mode == "encoder_only":
            p.requires_grad_(not is_readout)
        else:
            p.requires_grad_(train_all)
    for p in model.head.parameters():
        if mode == "readout_only":
            p.requires_grad_(True)
        elif mode == "encoder_only":
            p.requires_grad_(False)
        else:
            p.requires_grad_(train_all)


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


# ------------------------------------------------------------- instrumentation

def param_group_snapshot(model) -> dict[str, torch.Tensor]:
    """Flattened parameter states of the groups whose displacement is tracked."""
    ssm = model.temporal_model
    groups = {
        "head": torch.cat([p.detach().flatten() for p in model.head.parameters()]),
        "ssm_C": ssm.C.weight.detach().flatten(),
        "ssm_D": ssm.D.weight.detach().flatten(),
        "pooling": torch.cat([p.detach().flatten() for p in model.pooling.parameters()]),
        "gnn": torch.cat([p.detach().flatten() for p in model.graph_encoder.parameters()]),
    }
    if ssm.diagonal:
        groups["ssm_gen"] = torch.cat([ssm.A_raw.detach().flatten(), ssm.B.weight.detach().flatten()])
    else:
        groups["ssm_gen"] = ssm.B.weight.detach().flatten()
    return {k: v.cpu().clone() for k, v in groups.items()}


def param_group_deltas(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        k: float(torch.linalg.vector_norm(after[k].float() - before[k].float()).item())
        for k in before
    }


# ----------------------------------------------------------------- stage executor

def _run_episodes(trainers, episodes: int, mode: str, rehearsal_every: int = 4,
                  model=None, test_ds=None, probe_kwargs=None, capture_threshold=None,
                  probe_every: int = 5) -> tuple[bool, int, dict | None]:
    """``mode``:
      ``interleaved``: half of the episodes on the immediate env, half delayed;
      ``delayed_only``: all delayed;
      ``delayed_rehearsal``: run delayed with one immediate episode every
        ``rehearsal_every`` delayed episodes;
      ``delayed_capture``: delayed-only, probed every ``probe_every`` episodes;
        training stops at the first probe with delayed_intact >=
        ``capture_threshold`` (returns the capture info).
    Returns ``(captured, episodes_used, capture_info)``.
    """
    tr_imm, tr_del = trainers
    if mode == "interleaved":
        tr_imm.train(episodes // 2)
        tr_del.train(episodes - episodes // 2)
        return False, episodes, None
    if mode == "delayed_only":
        tr_del.train(episodes)
        return False, episodes, None
    if mode == "delayed_rehearsal":
        done = 0
        while done < episodes:
            burst = min(rehearsal_every, episodes - done)
            tr_del.train(burst)
            done += burst
            if done < episodes:
                tr_imm.train(1)
                done += 1
        return False, episodes, None
    if mode == "delayed_capture":
        done = 0
        while done < episodes:
            tr_del.train(probe_every)
            done += probe_every
            probe = quick_probe(
                model, test_ds,
                delay=probe_kwargs["delay"], marker_dim=probe_kwargs["marker_dim"],
                generic_blank=probe_kwargs["generic_blank"],
            )
            if probe["delayed_intact"] >= capture_threshold:
                return True, done, probe
        return False, episodes, None
    raise ValueError(f"unknown stage mode {mode!r}")


def _state_summary(model, test_ds, delay, marker_dim, generic_blank, seed, run_dir, tag) -> dict:
    _, summary = state_eval.run_state_content(
        model, test_ds, delay=delay, batch_size=50,
        marker_dim=marker_dim, generic_blank=generic_blank, seed=seed,
    )
    clean = {k: v for k, v in summary.items()
             if isinstance(v, (int, float)) and np.isfinite(v)}
    with (Path(run_dir) / f"state_content_{tag}.json").open("w") as f:
        import json
        json.dump(clean, f, indent=2, default=float)
    return clean


def run_coupling_arm(
    cfg: dict,
    datasets: dict,
    ckpt: dict,
    arm: str,
    run_dir: Path,
    seed: int,
    schedule_spec: dict,
) -> dict:
    """One arm on one starting checkpoint; writes run artifacts to
    ``run_dir`` and returns the summary."""
    import json

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
    tr_imm = Trainer(model, env_imm, optimizer, train_cfg)
    tr_del = Trainer(model, env_del, optimizer, train_cfg)
    tr_shuf = Trainer(model, env_shuf, optimizer, train_cfg) if env_shuf is not None else None

    # ---------------- pre-intervention analysis ----------------
    _, pre_summary = state_eval.run_state_content(
        model, test_ds, delay=delay, batch_size=50, marker_dim=marker_dim,
        generic_blank=generic_blank, seed=seed,
    )
    pre_summary = {k: v for k, v in pre_summary.items()
                   if isinstance(v, (int, float)) and np.isfinite(v)}
    snapshot_start = param_group_snapshot(model)

    rows: list[dict] = []
    ep_total = 0
    capture_info: dict | None = None

    def probe(tag: str) -> None:
        quick = quick_probe(model, test_ds, delay, marker_dim, generic_blank)
        rows.append({"phase": tag, "episodes_done": ep_total, **quick})

    probe("pre")

    # ---------------- arm-specific stage configs ----------------
    h5_stage3 = {"mode": "delayed_only", "episodes": int(schedule_spec.get("stage3", {}).get("episodes", 200))}
    interleaved4 = {"mode": "interleaved", "episodes": int(schedule_spec.get("stage4", {}).get("episodes", 200))}
    probe_kwargs = {"delay": delay, "marker_dim": marker_dim, "generic_blank": generic_blank}
    if arm == "control":
        stages = [{"mode": "interleaved", "episodes": h5_stage3["episodes"]}, dict(interleaved4)]
        freeze3, freeze4 = "all", "all"
    elif arm in ("treatment_a", "treatment_b", "h6_control"):
        stages = [dict(h5_stage3), dict(interleaved4)]
        freeze3, freeze4 = "readout_only", "all"
    elif arm == "h6_rehearsal":
        stages = [
            {"mode": "delayed_rehearsal", "episodes": h5_stage3["episodes"],
             "rehearsal_every": int(schedule_spec.get("rehearsal_every", 4))},
            dict(interleaved4),
        ]
        freeze3, freeze4 = "readout_only", "all"
    elif arm == "h6_freeze_readout":
        stages = [dict(h5_stage3), dict(interleaved4)]
        freeze3, freeze4 = "readout_only", "encoder_only"
    elif arm == "h7_capture_hold":
        stages = [
            {"mode": "delayed_capture", "episodes": h5_stage3["episodes"],
             "probe_every": int(schedule_spec.get("capture_probe_every", 5)),
             "threshold": float(schedule_spec.get("capture_threshold", 0.9))},
            dict(interleaved4),
        ]
        freeze3, freeze4 = "readout_only", "encoder_only"
    else:
        raise ValueError(f"unknown arm {arm!r}")

    for i, stage in enumerate(stages):
        set_trainables(model, freeze3 if i == 0 else freeze4)
        if arm == "treatment_b" and i == 0:
            captured, used, cap = _run_episodes(
                (tr_imm, tr_shuf), int(stage["episodes"]), "delayed_only",
            )
        else:
            captured, used, cap = _run_episodes(
                (tr_imm, tr_del), int(stage["episodes"]), stage["mode"],
                stage.get("rehearsal_every", 4),
                model=model, test_ds=test_ds, probe_kwargs=probe_kwargs,
                capture_threshold=stage.get("threshold"),
                probe_every=stage.get("probe_every", 5),
            )
        if captured and capture_info is None:
            capture_info = {"stage": i + 1, "episode_in_stage": used, **cap}
        ep_total += int(stage["episodes"])
        probe("post_stage" if i == 0 else "post_restore")

        tag = "stage3" if i == 0 else "stage4"
        state_summary = _state_summary(
            model, test_ds, delay, marker_dim, generic_blank, seed, run_dir, tag
        )
        if i == 0:
            stage3_summary = state_summary
            snapshot_after_stage3 = param_group_snapshot(model)
        post_summary = state_summary

    # ---------------- final battery + success rubric (pre-registered) ----------------
    battery = _final_battery(model, test_ds, delay, marker_dim, generic_blank, seed)
    snapshot_end = param_group_snapshot(model)
    dec_post = post_summary.get("decode_pre_response", np.nan)
    rubric = {
        "immediate_ok": bool(battery["immediate"] >= 0.90),
        "delayed_ok": bool(battery["delayed_intact"] >= 0.90),
        "subst_same_ok": bool(post_summary.get("subst_same_answer_accuracy", 0.0) >= 0.90),
        "subst_diff_follows": float(post_summary.get("subst_diff_follows", 0.0)),
        "zero_chance": bool(post_summary.get("subst_zero_accuracy", 1.0) <= 0.65),
        "shuffled_chance": bool(post_summary.get("subst_shuffled_accuracy", 1.0) <= 0.65),
        "decode_preserved": bool(dec_post >= 0.95),
        "state_norm_ratio_post_over_pre":
            float(post_summary.get("avf_mean_pre_response_norm", np.nan))
            / float(pre_summary.get("avf_mean_pre_response_norm", np.nan)),
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
        "stage3_analysis": stage3_summary,
        "post_analysis": post_summary,
        "battery": battery,
        "capture_info": capture_info,
        "param_deltas_after_stage3": param_group_deltas(snapshot_start, snapshot_after_stage3),
        "param_deltas_stage4": param_group_deltas(snapshot_after_stage3, snapshot_end),
        "rubric": rubric,
        "episodes_total": ep_total,
    }
    pd.DataFrame(rows).to_csv(run_dir / "training_probes.csv", index=False)
    with (run_dir / "arm_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary
