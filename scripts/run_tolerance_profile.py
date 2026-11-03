#!/usr/bin/env python
"""Tolerance profile: distance-to-boundary vs maintenance tolerance.

Report section 8t (H8t), pre-registered.  For each of the 12 captured delay-30
checkpoints, three measurements, no training:

  Part 1  geometry at capture (the 8l stage-A instrument, unchanged):
          m_behav  = abs_margin_mean / |w_r|           (behavioural boundary)
          m_decode = min class |probe proj| / |w_p|    (decoding boundary)
          disp_state, class_sep_state, state_norm, boundary_vs_mid, cos...
  Part 2  static displacement walk: theta(delta) = theta0 + delta * u for
          u in {u_grad (unit delayed-gradient at capture), u_rand (seeded
          unit random)} and delta on the factor-3 ladder
          {0.0005 ... 0.45}.  At every rung (pure translation): the full
          substitution battery, the state-content summary (subst_diff_follows,
          decode_pre_response, ...).  Thresholds at the 8q rubric criteria.
  Part 2b curvature ratio: one full-magnitude Adam step at capture, projected
          and restored exactly as in the 8s micro-replication; ratio =
          realised displacement / |J d_safe|.

  python scripts/run_tolerance_profile.py [--shard 0 --num-shards 12] [--merge]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from artificial_system.environment.graph_environment import (  # noqa: E402
    GraphClassificationEnvironment,
    make_generic_graph,
)
from artificial_system.evaluation import readout_geometry as rg  # noqa: E402
from artificial_system.evaluation import state_content as sc  # noqa: E402
from artificial_system.evaluation.state_content import ridge_decode_cv  # noqa: E402
from artificial_system.experiments import coupling_rescue as cr  # noqa: E402
from artificial_system.models.gnn_ssm import build_model  # noqa: E402
from artificial_system.training.losses import IGNORE_INDEX, classification_loss  # noqa: E402

import run_jacobian_projected as rjp  # noqa: E402

LADDER = [0.0005, 0.0015, 0.005, 0.015, 0.05, 0.15, 0.45]
captured_checkpoints = rjp.captured_checkpoints
VAR_FLOOR = 0.01  # 8o variance floor


def state_dispersion(model, test_ds, delay, marker_dim, generic_blank) -> dict:
    """Within-class dispersion (8o variance floor) and class separation, state units."""
    model.eval()
    st = sc.capture_trial_states(model, test_ds, delay=delay, batch_size=50,
                                 marker_dim=marker_dim, generic_blank=generic_blank)
    X = st["pre_response"].detach().float().numpy()
    y = st["_label"].numpy().astype(int)
    mu0, mu1 = X[y == 0].mean(0), X[y == 1].mean(0)
    var = []
    for c in (0, 1):
        Xc = X[y == c]
        if len(Xc) > 1:
            var.append(Xc.var(0, ddof=1))
    pooled = np.mean(var, axis=0) if var else np.zeros(X.shape[1])
    pooled = np.maximum(pooled, VAR_FLOOR)
    return {
        "disp_state": float(np.sqrt(pooled.mean())),
        "class_sep_state": float(np.linalg.norm(mu1 - mu0)),
        "state_norm_mean_geo": float(np.linalg.norm(X, axis=1).mean()),
    }


def replay_first_episode_gradient(model, datasets, task_cfg, ck, train_cfg) -> torch.Tensor:
    """The delayed objective's gradient at capture (one episode, as 8n/8s)."""
    seed = int(ck["seed"])
    batch_size = int(train_cfg.get("batch_size", 16))
    seq_len = int(train_cfg.get("sequence_length", 32))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    env = GraphClassificationEnvironment(datasets["train"], task_cfg, seed=seed * 2000 + 2)
    rng_states = ck.get("rng_states") or {}
    if "env_del" in rng_states:
        env.rng.set_state(rng_states["env_del"])
    model.train()
    obs = env.reset(batch_size)
    state = model.initial_state(batch_size)
    losses = []
    for _ in range(seq_len):
        logits, state, _aux = model.step(obs, state)
        targets = env.current_targets()
        losses.append(classification_loss(logits, targets, IGNORE_INDEX))
        with torch.no_grad():
            actions = logits.argmax(-1)
        obs, _r, _d, _i = env.step(actions)
    loss = torch.stack(losses).mean()
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    enc = [p for p in model.parameters() if p.requires_grad]
    g = torch.cat([p.grad.reshape(-1) for p in enc]).detach()
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    return g


def curvature_block(model, enc, theta0, datasets, task_cfg, ck, train_cfg,
                    probe_ds, s_cap, delay, marker_dim, generic_blank) -> dict:
    """One full-magnitude Adam step at capture, projected and restored with J
    evaluated at theta_pre -- exactly the 8s micro-replication recipe."""
    g_vec = replay_first_episode_gradient(model, datasets, task_cfg, ck, train_cfg)
    off = 0
    for p in enc:
        n = p.numel()
        p.grad = g_vec[off:off + n].view_as(p).clone()
        off += n
    # J at theta_pre (BEFORE the step) -- the registered 8s recipe
    J = rjp.response_state_jacobian(model, probe_ds, delay, marker_dim,
                                    generic_blank, enc).detach()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    optimizer.step()
    with torch.no_grad():
        d_theta = torch.cat([(p.detach() - t).reshape(-1) for p, t in zip(enc, theta0)])
        Jd = J @ d_theta
        U = torch.linalg.pinv(J, rcond=1e-6)
        d_safe = d_theta - U @ Jd
        d_safe = d_safe * (d_theta.norm() / d_safe.norm())
        lin_pred = float((J @ d_safe).norm())   # post-restoration, as in 8s
        off = 0
        for p, t in zip(enc, theta0):
            n = p.numel()
            p.copy_(t + d_safe[off:off + n].view_as(p))
            off += n
    disp1 = rjp.probe_displacement(model, probe_ds, s_cap, delay, marker_dim, generic_blank)
    with torch.no_grad():
        off = 0
        for p, t in zip(enc, theta0):
            n = p.numel()
            p.copy_(t)
            off += n
    return {
        "adam_norm": float(d_theta.norm()),
        "lin_pred": lin_pred,
        "disp_step1": disp1,
        "curvature_ratio": disp1 / lin_pred if lin_pred > 0 else float("nan"),
    }


def run_one(ck_path: Path, run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    seed = int(ck["seed"])
    delay = int(ck.get("delay", 30))
    task_cfg = dict(cfg.get("task", {}))
    task_cfg["delay"] = delay
    marker_dim = int(task_cfg.get("marker_dim", 2))
    train_cfg = dict(cfg.get("training", {}))

    datasets = cr.build_datasets_from_cfg(cfg)
    test_ds = datasets["test"]

    model = build_model(
        cfg, node_feature_dim=int(cfg.get("graph", {}).get("node_feature_dim", 2)),
        edge_feature_dim=0, num_actions=2, num_nodes=None,
    )
    model.load_state_dict(ck["model_state"])
    cr.set_trainables(model, "encoder_only")
    enc = [p for p in model.parameters() if p.requires_grad]

    generic_blank = None
    if task_cfg.get("blank_mode", "no_markers") == "generic":
        generic_blank = make_generic_graph(
            int(task_cfg.get("num_nodes", 100)),
            int(cfg.get("graph", {}).get("node_feature_dim", 2)),
            seed * 2000 + 2,
        )

    # ---------------- Part 1: geometry at capture ----------------
    model.eval()
    g1 = rg.analyse(model, test_ds, delay, marker_dim, generic_blank, seed)
    disp = state_dispersion(model, test_ds, delay, marker_dim, generic_blank)
    part1 = {
        "m_behav": g1["abs_margin_mean"] / g1["readout_norm"] if g1["readout_norm"] else float("nan"),
        "m_decode": min(abs(g1["proj_probe_class0"]), abs(g1["proj_probe_class1"])) / g1["probe_norm"]
        if g1["probe_norm"] else float("nan"),
        **{k: g1[k] for k in ("abs_margin_mean", "readout_norm", "probe_norm",
                              "class_margin_separation", "boundary_vs_mid",
                              "cos_probe_readout", "readout_accuracy",
                              "decode_accuracy", "state_norm_mean")},
        **disp,
    }

    # ---------------- directions ----------------
    theta0 = [p.detach().clone() for p in enc]
    g_vec = replay_first_episode_gradient(model, datasets, task_cfg, ck, train_cfg)
    grad_norm0 = float(g_vec.norm())
    u_grad = g_vec / grad_norm0
    gen = torch.Generator().manual_seed(seed * 77 + 3)
    u_rand = torch.randn(g_vec.numel(), generator=gen)
    u_rand = u_rand / u_rand.norm()

    # group slices for per-group displacement recording
    sizes = [p.numel() for p in enc]
    group_names = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            group_names.append(n.split(".")[0])

    def group_norms(vec: torch.Tensor) -> dict:
        out, off = {}, 0
        for name, n in zip(group_names, sizes):
            out[name] = float(vec[off:off + n].norm())
            off += n
        return out

    # ---------------- Part 2: static displacement walk ----------------
    cap_summary = cr._state_summary(
        model, test_ds, delay, marker_dim, generic_blank, seed, run_dir, "cap")
    walk_rows = []
    for dname, u in (("u_grad", u_grad), ("u_rand", u_rand)):
        for delta in LADDER:
            with torch.no_grad():
                off = 0
                for p, t in zip(enc, theta0):
                    n = p.numel()
                    p.copy_(t + delta * u[off:off + n].view_as(p))
                    off += n
            model.eval()
            rung_summary = cr._state_summary(
                model, test_ds, delay, marker_dim, generic_blank, seed,
                run_dir, f"{dname}_{delta}")
            battery = cr._final_battery(
                model, test_ds, delay, marker_dim, generic_blank, seed)
            gn = group_norms(delta * u)
            walk_rows.append({
                "direction": dname, "delta": delta,
                "delayed_intact": battery["delayed_intact"],
                "delayed_reset": battery["delayed_reset"],
                "immediate": battery["immediate"],
                "subst_diff_follows": rung_summary.get("subst_diff_follows"),
                "subst_same": rung_summary.get("subst_same_answer_accuracy"),
                "decode_pre_response": rung_summary.get("decode_pre_response"),
                "d_total": float(delta),
                **{f"d_{k}": v for k, v in gn.items()},
            })
            print(f"    {dname} delta={delta:.4f} delayed={battery['delayed_intact']:.3f} "
                  f"follows={rung_summary.get('subst_diff_follows')} "
                  f"decode={rung_summary.get('decode_pre_response')}", flush=True)
    with torch.no_grad():
        off = 0
        for p, t in zip(enc, theta0):
            n = p.numel()
            p.copy_(t)
            off += n

    pd.DataFrame(walk_rows).to_csv(run_dir / "walk_log.csv", index=False)

    # ---------------- Part 2b: curvature ratio (8s micro-replication) ----------------
    probe_ds = type(test_ds)(
        graphs=list(test_ds.graphs[:16]), labels=test_ds.labels[:16].clone())
    model.eval()
    st_cap = sc.capture_trial_states(model, probe_ds, delay=delay, batch_size=16,
                                     marker_dim=marker_dim, generic_blank=generic_blank)
    s_cap = st_cap["pre_response"].detach().float()
    curvature = curvature_block(model, enc, theta0, datasets, task_cfg, ck, train_cfg,
                                probe_ds, s_cap, delay, marker_dim, generic_blank)

    summary = {
        "arm": "tolerance_profile",
        "seed": seed,
        "delay": int(delay),
        "source_checkpoint": str(ck_path.relative_to(ROOT)),
        "part1": part1,
        "grad_norm0": grad_norm0,
        "curvature": curvature,
        "walk_rungs": len(walk_rows),
    }
    with (run_dir / "tolerance_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary


def _merge(out_dir: Path) -> None:
    merged: dict = {}
    for f in sorted(out_dir.glob("*__*/tolerance_summary.json")):
        merged[f.parent.name] = json.loads(f.read_text())
    (out_dir / "summary.json").write_text(json.dumps(merged, indent=2, default=float))
    print(f"merged {len(merged)} runs -> {out_dir / 'summary.json'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/delay30_tolerance")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=12)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge:
        _merge(out_dir)
        return 0

    ckpts = captured_checkpoints()
    mine = [c for i, c in enumerate(ckpts) if i % args.num_shards == args.shard]
    print(f"[shard {args.shard}/{args.num_shards}] {len(mine)} of {len(ckpts)} checkpoints",
          flush=True)

    failures = 0
    for tag, ck_path in mine:
        run_dir = out_dir / f"{tag}__tolerance"
        if (run_dir / "tolerance_summary.json").exists():
            print(f"[skip] {tag} (complete)", flush=True)
            continue
        print(f"[run ] {tag} -> {run_dir}", flush=True)
        try:
            s = run_one(ck_path, run_dir)
            p1, c = s["part1"], s["curvature"]
            print(f"      m_behav={p1['m_behav']:.4f} m_decode={p1['m_decode']:.4f} "
                  f"disp={p1['disp_state']:.4f} curvature={c['curvature_ratio']:.1f}x",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"      ERROR in {tag}: {exc}", flush=True)
            raise
    _merge(out_dir)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
