#!/usr/bin/env python
"""Direction at full step size, applied to the realised update: post-Adam step
projection.

Report section 8s (H8s).  Same 12 captured checkpoints, same exposure as the
unconstrained reference (report 8n ``delayed_only``: 100 delayed episodes at
delay 30, saved RNG stream, frozen readout), same J as section 8r.

Per step:
  1. J is evaluated at theta_pre (before the update);
  2. Adam steps with the *raw* clipped delayed gradient (identical to 8n);
  3. the realised step is projected:
         d_safe = d_theta - pinv(J) (J d_theta)        (rcond 1e-6)
  4. magnitude restoration: d_safe is rescaled to |d_theta| (the unconstrained
     magnitude) -- the encoder moves exactly as far as 8n, in the protected
     direction;
  5. if |d_safe| < 1e-12 (the step is entirely state-changing) the step is
     rejected (theta unchanged).

Because the projection is applied after the optimiser, the realised
displacement is state-null to first order at theta_pre by construction -- the
8r failure channel (Adam reintroducing forbidden directions) is closed.
Recorded per step: raw gradient norm, Adam's realised update norm, the
projection ratio |d_safe|/|d_theta|, the nulling residual |J d_safe|/|J d_theta|,
the rejection flag, and the state displacement after *every* update.

  python scripts/run_step_projected.py [--shard 0 --num-shards 12] [--merge]
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
from artificial_system.evaluation import state_content as sc  # noqa: E402
from artificial_system.experiments import coupling_rescue as cr  # noqa: E402
from artificial_system.models.gnn_ssm import build_model  # noqa: E402
from artificial_system.training.losses import IGNORE_INDEX, classification_loss  # noqa: E402

import run_jacobian_projected as rjp  # noqa: E402

PROBE_N = rjp.PROBE_N
PROBE_AT = rjp.PROBE_AT
RCOND = rjp.RCOND
captured_checkpoints = rjp.captured_checkpoints
response_state_jacobian = rjp.response_state_jacobian
probe_displacement = rjp.probe_displacement

REJECT_EPS = 1e-12


def run_one(ck_path: Path, run_dir: Path, episodes: int) -> dict:
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
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    batch_size = int(train_cfg.get("batch_size", 16))
    seq_len = int(train_cfg.get("sequence_length", 32))

    datasets = cr.build_datasets_from_cfg(cfg)
    test_ds = datasets["test"]
    from artificial_system.data.graph_tasks import ConnectivityDataset
    probe_ds = ConnectivityDataset(
        graphs=list(test_ds.graphs[:PROBE_N]), labels=test_ds.labels[:PROBE_N].clone(),
    )

    model = build_model(
        cfg, node_feature_dim=int(cfg.get("graph", {}).get("node_feature_dim", 2)),
        edge_feature_dim=0, num_actions=2, num_nodes=None,
    )
    model.load_state_dict(ck["model_state"])
    cr.set_trainables(model, "encoder_only")
    params = [p for p in model.parameters() if p.requires_grad]

    generic_blank = None
    if task_cfg.get("blank_mode", "no_markers") == "generic":
        generic_blank = make_generic_graph(
            int(task_cfg.get("num_nodes", 100)),
            int(cfg.get("graph", {}).get("node_feature_dim", 2)),
            seed * 2000 + 2,
        )

    # captured reference states on the probe batch
    model.eval()
    st_cap = sc.capture_trial_states(
        model, probe_ds, delay=delay, batch_size=PROBE_N,
        marker_dim=marker_dim, generic_blank=generic_blank,
    )
    s_cap = st_cap["pre_response"].detach().float()

    env = GraphClassificationEnvironment(datasets["train"], task_cfg, seed=seed * 2000 + 2)
    rng_states = ck.get("rng_states") or {}
    if "env_del" in rng_states:
        env.rng.set_state(rng_states["env_del"])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    pre_summary = cr._state_summary(
        model, test_ds, delay, marker_dim, generic_blank, seed, run_dir, "pre"
    )
    snap_start = cr.param_group_snapshot(model)

    rows: list[dict] = []
    step_rows: list[dict] = []

    def probe(phase: str, done: int) -> None:
        quick = cr.quick_probe(model, test_ds, delay, marker_dim, generic_blank)
        rows.append({"phase": phase, "episodes_done": done, **quick})

    probe("pre", 0)
    model.train()

    for ep in range(1, episodes + 1):
        # ---- one episode, replicating Trainer.train exactly up to the gradient
        obs = env.reset(batch_size)
        state = model.initial_state(batch_size)
        losses: list[torch.Tensor] = []
        for _ in range(seq_len):
            logits, state, _aux = model.step(obs, state)
            targets = env.current_targets()
            losses.append(classification_loss(logits, targets, IGNORE_INDEX))
            with torch.no_grad():
                actions = logits.argmax(-1)
            obs, _rewards, _done, _info = env.step(actions)

        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        g_vec = torch.cat([p.grad.reshape(-1) for p in params]).detach()
        raw_norm = float(g_vec.norm())

        # J at theta_pre (requires grad mode; own forward graph)
        J = response_state_jacobian(
            model, probe_ds, delay, marker_dim, generic_blank, params
        ).detach()

        theta_pre = [p.detach().clone() for p in params]

        # Adam steps with the raw clipped gradient -- identical to 8n
        optimizer.step()

        with torch.no_grad():
            d_theta = torch.cat([
                (p.detach() - t).reshape(-1) for p, t in zip(params, theta_pre)
            ])
            d_norm = float(d_theta.norm())

            Jd = J @ d_theta
            jd_norm = float(Jd.norm())
            U = torch.linalg.pinv(J, rcond=RCOND)              # [P, 128]
            d_safe = d_theta - U @ Jd
            safe_norm = float(d_safe.norm())

            rejected = safe_norm < REJECT_EPS
            if rejected:
                # entirely state-changing: reject the step (theta unchanged)
                for p, t in zip(params, theta_pre):
                    p.data.copy_(t)
                ratio = 0.0
                restored_norm = 0.0
                resid_ratio = float("nan")
            else:
                ratio = safe_norm / d_norm if d_norm > 0 else float("nan")
                # magnitude restoration to the unconstrained norm
                d_safe = d_safe * (d_norm / safe_norm)
                restored_norm = float(d_safe.norm())
                resid_ratio = float((J @ d_safe).norm() / jd_norm) if jd_norm > 0 else float("nan")
                off = 0
                for p, t in zip(params, theta_pre):
                    n = p.numel()
                    p.data.copy_(t + d_safe[off:off + n].view_as(p))
                    off += n

        disp = probe_displacement(model, probe_ds, s_cap, delay, marker_dim, generic_blank)
        step_rows.append({
            "episode": ep,
            "raw_grad_norm": raw_norm,
            "adam_update_norm": d_norm,
            "projection_ratio": ratio,
            "restored_norm": restored_norm,
            "state_residual_ratio": resid_ratio,
            "rejected": bool(rejected),
            "displacement": disp,
        })

        if ep in PROBE_AT:
            probe(f"ep{ep}", ep)
            pd.DataFrame(rows).to_csv(run_dir / "training_probes.csv", index=False)

    post_summary = cr._state_summary(
        model, test_ds, delay, marker_dim, generic_blank, seed, run_dir, "post"
    )
    battery = cr._final_battery(model, test_ds, delay, marker_dim, generic_blank, seed)
    snap_end = cr.param_group_snapshot(model)
    rubric = cr.success_rubric(pre_summary, post_summary, battery)

    pd.DataFrame(step_rows).to_csv(run_dir / "step_log.csv", index=False)
    ratios = [r["projection_ratio"] for r in step_rows]
    ratios_f = [r for r in ratios if r == r]
    d = cr.param_group_deltas(snap_start, snap_end)

    summary = {
        "arm": "step_projected",
        "seed": seed,
        "delay": int(delay),
        "source_checkpoint": str(ck_path.relative_to(ROOT)),
        "captured_from": ck_path.parent.name,
        "pre_analysis": pre_summary,
        "post_analysis": post_summary,
        "battery": battery,
        "episodes_run": int(episodes),
        "param_deltas": d,
        "rubric": rubric,
        "projection_ratio_mean": float(np.mean(ratios_f)) if ratios_f else float("nan"),
        "projection_ratio_min": float(np.min(ratios_f)) if ratios_f else float("nan"),
        "projection_ratio_max": float(np.max(ratios_f)) if ratios_f else float("nan"),
        "rejected_steps": int(sum(1 for r in step_rows if r["rejected"])),
        "adam_update_norm_mean": float(np.mean([r["adam_update_norm"] for r in step_rows])),
        "raw_grad_norm_mean": float(np.mean([r["raw_grad_norm"] for r in step_rows])),
        "displacement_mean": float(np.mean([r["displacement"] for r in step_rows])),
        "displacement_max": float(np.max([r["displacement"] for r in step_rows])),
        "displacement_final": float(step_rows[-1]["displacement"]) if step_rows else float("nan"),
    }
    pd.DataFrame(rows).to_csv(run_dir / "training_probes.csv", index=False)
    with (run_dir / "arm_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary


def _merge(out_dir: Path) -> None:
    merged: dict = {}
    for f in sorted(out_dir.glob("*__*/arm_summary.json")):
        merged[f.parent.name] = json.loads(f.read_text())
    (out_dir / "summary.json").write_text(json.dumps(merged, indent=2, default=float))
    print(f"merged {len(merged)} runs -> {out_dir / 'summary.json'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/delay30_step_proj")
    ap.add_argument("--episodes", type=int, default=100)
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
    print(f"[shard {args.shard}/{args.num_shards}] {len(mine)} of {len(ckpts)} "
          f"checkpoints, {args.episodes} episodes", flush=True)

    failures = 0
    for tag, ck_path in mine:
        run_dir = out_dir / f"{tag}__step_projected"
        if (run_dir / "arm_summary.json").exists():
            print(f"[skip] {tag} (complete)", flush=True)
            continue
        print(f"[run ] {tag} -> {run_dir}", flush=True)
        try:
            s = run_one(ck_path, run_dir, args.episodes)
            r, b = s["rubric"], s["battery"]
            print(f"      delayed={b['delayed_intact']:.3f} "
                  f"follows={r['subst_diff_follows']:.3f} "
                  f"decode={s['post_analysis'].get('decode_pre_response', float('nan')):.3f} "
                  f"d_gnn={s['param_deltas']['gnn']:.3f} "
                  f"ratio={s['projection_ratio_mean']:.3f} "
                  f"disp_final={s['displacement_final']:.3f} "
                  f"rej={s['rejected_steps']} "
                  f"ALL={'PASS' if r['all_criteria'] else 'fail'}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"      ERROR in {tag}: {exc}", flush=True)
            raise
    _merge(out_dir)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
