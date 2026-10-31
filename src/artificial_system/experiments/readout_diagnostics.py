"""Readout-path convergence diagnostics for one failing checkpoint (§8d).

Hand-rolled readout-only delayed training loop that additionally measures,
per episode and learning rate and restart:

- delayed / immediate loss at the point each gradient is formed;
- gradient norms per trainable readout group (head, SSM-C, SSM-D);
- cosine similarity between the immediate-loss gradient and the delayed-loss
  gradient (conflict diagnostic);
- logit margins at response steps;
- immediate/delayed probing every ``probe_every`` episodes.

The starting checkpoint is only read; all runs optimise in-memory copies.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from artificial_system.environment.graph_environment import GraphClassificationEnvironment
from artificial_system.evaluation import graph_classification as graph_eval
from artificial_system.training.losses import IGNORE_INDEX, classification_loss

READOUT_SSM_PREFIXES = ("C.", "D.")


def _trainable_params(model) -> list[torch.nn.Parameter]:
    ssm = model.temporal_model
    params = [p for p in model.head.parameters()]
    params += [p for name, p in ssm.named_parameters() if name.startswith(READOUT_SSM_PREFIXES)]
    return params


def _flat(params, grads) -> torch.Tensor:
    return torch.cat([g.detach().flatten() for g in grads])


def _episode(
    model,
    env: GraphClassificationEnvironment,
    batch_size: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor | None, float]:
    """One env trial; returns (loss, [margin | None], mean margin)."""
    obs = env.reset(batch_size)
    state = model.initial_state(batch_size)
    losses: list[torch.Tensor] = []
    margins: list[torch.Tensor] = []
    total_margin = 0.0
    for _ in range(seq_len):
        logits, state, _ = model.step(obs, state)
        targets = env.current_targets()
        losses.append(classification_loss(logits, targets, IGNORE_INDEX))
        active = targets != IGNORE_INDEX
        if bool(active.any()):
            m = (logits.max(-1).values - logits.min(-1).values).detach()
            margins.append(m[torch.as_tensor(active.nonzero(as_tuple=True)[0])])
        obs, _, _, _ = env.step(logits.argmax(-1))
    loss = torch.stack(losses).mean()
    if margins:
        total_margin = float(torch.stack(margins).mean())
    return loss, (torch.stack(margins).mean() if margins else None), total_margin


def _grad_of(loss: torch.Tensor, params) -> tuple[list[torch.Tensor], float]:
    grads = torch.autograd.grad(
        loss, [p for p in params if p.requires_grad], allow_unused=True, retain_graph=False
    )
    gs = [g if g is not None else torch.zeros_like(p)
          for g, p in zip(grads, [p for p in params if p.requires_grad])]
    return gs, float(sum(float(torch.linalg.vector_norm(g)) for g in gs))


@torch.no_grad()
def _probe(model, test_ds, delay, marker_dim, generic_blank):
    imm = graph_eval.probe_dataset(
        model, test_ds, delay=0, batch_size=32, marker_dim=marker_dim
    )["accuracy"]
    del_probe = graph_eval.probe_delayed_intact_vs_reset(
        model, test_ds, delay=delay, batch_size=32,
        marker_dim=marker_dim, generic_blank=generic_blank,
    )
    return imm, del_probe["delayed_intact"], del_probe["delayed_state_reset"]


# ---------------------------------------------- encoder-level conflict (§8f)

ENCODER_GROUPS = ("gnn", "pooling", "ssm_gen")


def _group_params(model, group: str) -> list[torch.nn.Parameter]:
    ssm = model.temporal_model
    if group == "gnn":
        return list(model.graph_encoder.parameters())
    if group == "pooling":
        return list(model.pooling.parameters())
    if group == "ssm_gen":
        return [ssm.A_raw, ssm.B.weight] if ssm.diagonal else [ssm.B.weight]
    raise ValueError(group)


def run_encoder_gradient_diagnostics(
    cfg: dict,
    datasets: dict,
    ckpt: dict,
    run_dir: Path,
    seed: int,
    episodes: int = 200,
    probe_every: int = 5,
    grad_clip: float = 1.0,
) -> dict:
    """§8f: per-group immediate-vs-delayed gradient conflict during the H7
    stage-4 regime (frozen readout, joint episodes), on the *captured*
    checkpoint of the failing source.

    Per episode: delayed and immediate losses; per parameter group (gnn,
    pooling, ssm_gen) the gradient norms ||g_del||, ||g_imm||, cosine
    cos(g_del, g_imm), and the combined norm ||g_del + g_imm||; delayed logit
    margins; behavioural probes every ``probe_every`` episodes (the cadence
    that revealed the ~20-episode oscillation). The applied update alternates
    delayed (even episodes) / immediate (odd) exactly like interleaved stage-4
    training, with the readout frozen throughout.
    """
    import json

    from artificial_system.models.gnn_ssm import build_model

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    task_cfg = cfg.get("task", {})
    train_cfg = dict(cfg.get("training", {}))
    delay = int(task_cfg.get("delay", 3))
    marker_dim = int(task_cfg.get("marker_dim", 2))
    batch = int(train_cfg.get("batch_size", 16))
    seq_len = int(train_cfg.get("sequence_length", 32))

    model = build_model(
        cfg, node_feature_dim=int(cfg.get("graph", {}).get("node_feature_dim", 2)),
        edge_feature_dim=0, num_actions=2, num_nodes=None,
    )
    model.load_state_dict(ckpt["model_state"])
    # frozen readout, live representation (H7 stage-4 regime)
    from artificial_system.experiments.coupling_rescue import set_trainables
    set_trainables(model, "encoder_only")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=float(train_cfg.get("learning_rate", 1e-3)))

    env_del = GraphClassificationEnvironment(datasets["train"], task_cfg, seed=seed * 2000 + 2)
    imm_cfg = dict(task_cfg)
    imm_cfg["delay"] = 0
    env_imm = GraphClassificationEnvironment(datasets["train"], imm_cfg, seed=seed * 2000 + 1)
    # continue the env RNG streams from the capture point (exact regime
    # reconstruction when rng_states are present in the checkpoint)
    rng_states = ckpt.get("rng_states")
    if rng_states is not None:
        env_del.rng.set_state(rng_states["env_del"])
        env_imm.rng.set_state(rng_states["env_imm"])
    generic_blank = None
    if task_cfg.get("blank_mode", "no_markers") == "generic":
        from artificial_system.environment.graph_environment import make_generic_graph
        generic_blank = make_generic_graph(
            int(task_cfg.get("num_nodes", 100)),
            int(cfg.get("graph", {}).get("node_feature_dim", 2)),
            seed * 2000 + 2,
        )

    groups = {g: _group_params(model, g) for g in ENCODER_GROUPS}

    def group_grads(loss):
        return torch.autograd.grad(loss, trainable, retain_graph=False, allow_unused=True)

    rows: list[dict] = []
    for ep in range(int(episodes)):
        model.train()
        for p in trainable:
            p.grad = None

        loss_d, margin_d, _ = _episode(model, env_del, batch, seq_len)
        g_del = group_grads(loss_d)
        loss_i, _, _ = _episode(model, env_imm, batch, seq_len)
        g_imm = group_grads(loss_i)

        row: dict = {"episode": ep, "loss_delayed": float(loss_d), "loss_immediate": float(loss_i),
                     "margin_delayed": float(margin_d) if margin_d is not None else float("nan")}
        for gname in ENCODER_GROUPS:
            idx = [j for j, p in enumerate(trainable) if any(p is q for q in groups[gname])]
            gd = [g_del[j] if g_del[j] is not None else torch.zeros_like(trainable[j]) for j in idx]
            gi = [g_imm[j] if g_imm[j] is not None else torch.zeros_like(trainable[j]) for j in idx]
            flat_d = torch.cat([g.flatten() for g in gd])
            flat_i = torch.cat([g.flatten() for g in gi])
            row[f"{gname}_gdel_norm"] = float(torch.linalg.vector_norm(flat_d))
            row[f"{gname}_gimm_norm"] = float(torch.linalg.vector_norm(flat_i))
            row[f"{gname}_cos"] = float(
                torch.nn.functional.cosine_similarity(flat_d, flat_i, dim=0)
            )
            row[f"{gname}_combined_norm"] = float(torch.linalg.vector_norm(flat_d + flat_i))

        # apply the stage-4-equivalent update: alternate delayed/immediate
        which = g_del if ep % 2 == 0 else g_imm
        for p, g in zip(trainable, which):
            p.grad = (g.clone() if g is not None else torch.zeros_like(p))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()

        if ep % probe_every == 0 or ep == int(episodes) - 1:
            imm, del_i, del_r = _probe(model, datasets["test"], delay, marker_dim, generic_blank)
            row.update({"probe_immediate": imm, "probe_delayed_intact": del_i,
                        "probe_delayed_reset": del_r})
        else:
            row.update({"probe_immediate": float("nan"), "probe_delayed_intact": float("nan"),
                        "probe_delayed_reset": float("nan")})
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "encoder_gradients.csv", index=False)
    tail = df.tail(len(df) // 4)
    summary = {
        "episodes": int(episodes),
        "cos_mean": {g: float(df[f"{g}_cos"].mean()) for g in ENCODER_GROUPS},
        "cos_tail_mean": {g: float(tail[f"{g}_cos"].mean()) for g in ENCODER_GROUPS},
        "cos_frac_negative": {g: float((df[f"{g}_cos"] < 0).mean()) for g in ENCODER_GROUPS},
        "gdel_norm_mean": {g: float(df[f"{g}_gdel_norm"].mean()) for g in ENCODER_GROUPS},
        "gimm_norm_mean": {g: float(df[f"{g}_gimm_norm"].mean()) for g in ENCODER_GROUPS},
        "delayed_final": float(df["probe_delayed_intact"].iloc[-1]),
        "immediate_final": float(df["probe_immediate"].iloc[-1]),
    }
    (run_dir / "encoder_gradients_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def run_readout_diagnostics(
    cfg: dict,
    datasets: dict,
    ckpt: dict,
    run_dir: Path,
    seed: int,
    episodes: int = 100,
    learning_rate: float = 1e-3,
    cos_every: int = 1,
    probe_every: int = 10,
    grad_clip: float | None = None,
    restart_offset: int = 0,
) -> dict:
    import json

    from artificial_system.models.gnn_ssm import build_model

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    task_cfg = cfg.get("task", {})
    train_cfg = dict(cfg.get("training", {}))
    delay = int(task_cfg.get("delay", 3))
    marker_dim = int(task_cfg.get("marker_dim", 2))
    batch = int(train_cfg.get("batch_size", 16))
    seq_len = int(train_cfg.get("sequence_length", 32))
    if grad_clip is None:
        grad_clip = float(train_cfg.get("grad_clip", 0.0))

    model = build_model(
        cfg, node_feature_dim=int(cfg.get("graph", {}).get("node_feature_dim", 2)),
        edge_feature_dim=0, num_actions=2, num_nodes=None,
    )
    model.load_state_dict(ckpt["model_state"])
    for p in model.parameters():
        p.requires_grad_(False)
    params = _trainable_params(model)
    for p in params:
        p.requires_grad_(True)

    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0.0)

    env_del = GraphClassificationEnvironment(datasets["train"], task_cfg, seed=seed * 2000 + 2)
    imm_cfg = dict(task_cfg)
    imm_cfg["delay"] = 0
    env_imm = GraphClassificationEnvironment(datasets["train"], imm_cfg, seed=seed * 2000 + 1)
    # restart variation: data-order seeds; the checkpoint itself is a copy
    env_del.rng.manual_seed(seed * 2000 + 2 + restart_offset)
    env_imm.rng.manual_seed(seed * 2000 + 1 + restart_offset)

    generic_blank = None
    if task_cfg.get("blank_mode", "no_markers") == "generic":
        from artificial_system.environment.graph_environment import make_generic_graph
        generic_blank = make_generic_graph(
            int(task_cfg.get("num_nodes", 100)),
            int(cfg.get("graph", {}).get("node_feature_dim", 2)),
            seed * 2000 + 2,
        )

    rows: list[dict] = []
    for ep in range(int(episodes)):
        model.train()
        for p in params:
            p.grad = None
        loss_d, margin_d, _ = _episode(model, env_del, batch, seq_len)
        g_del, norm_del = _grad_of(loss_d, params)

        # immediate gradient (diagnostic only — never applied here)
        loss_i, _, _ = _episode(model, env_imm, batch, seq_len)
        g_imm, norm_imm = _grad_of(loss_i, params)

        if ep % cos_every == 0:
            v1 = torch.cat([g.flatten() for g in g_del])
            v2 = torch.cat([g.flatten() for g in g_imm])
            cos = float(torch.nn.functional.cosine_similarity(v1, v2, dim=0))
        else:
            cos = float("nan")

        # apply the DELAYED gradient (readout-only delayed training protocol)
        for p, g in zip([q for q in params if q.requires_grad], g_del):
            p.grad = g.clone()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step()

        if ep % probe_every == 0 or ep == episodes - 1:
            imm, del_i, del_r = _probe(model, test_ds=datasets["test"], delay=delay,
                                       marker_dim=marker_dim, generic_blank=generic_blank)
        else:
            imm = del_i = del_r = float("nan")
        rows.append({
            "lr": learning_rate,
            "episode": ep,
            "loss_delayed": float(loss_d),
            "loss_immediate": float(loss_i),
            "grad_norm_head": norm_del,
            "grad_norm_imm_imm": norm_imm,
            "grad_cos_imm_delayed": cos,
            "margin_delayed": float(margin_d) if margin_d is not None else float("nan"),
            "probe_immediate": imm,
            "probe_delayed_intact": del_i,
            "probe_delayed_reset": del_r,
        })

    df = pd.DataFrame(rows)
    csv_path = run_dir / f"diag_lr{learning_rate:g}_seed{seed}.csv"
    df.to_csv(csv_path, index=False)
    tail = df.tail(max(len(df) // 4, 1))
    summary = {
        "lr": learning_rate,
        "seed": seed,
        "episodes": int(episodes),
        "loss_delayed_initial": float(df["loss_delayed"].iloc[0]),
        "loss_delayed_final": float(df["loss_delayed"].iloc[-1]),
        "cos_imm_delayed_mean_tail": float(tail["grad_cos_imm_delayed"].mean(skipna=True)),
        "cos_imm_delayed_mean_all": float(df["grad_cos_imm_delayed"].mean(skipna=True)),
        "delayed_final": float(df["probe_delayed_intact"].iloc[-1]),
        "immediate_final": float(df["probe_immediate"].iloc[-1]),
        "grad_norm_tail_mean": float(tail["grad_norm_head"].mean(skipna=True)),
    }
    (run_dir / f"diag_lr{learning_rate:g}_seed{seed}_summary.json").write_text(
        json.dumps(summary, indent=2, default=float)
    )
    return summary
