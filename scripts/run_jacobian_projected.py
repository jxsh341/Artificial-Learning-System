#!/usr/bin/env python
"""Direction at full step size: state-Jacobian-projected delayed updates.

Report section 8r (H8r).  Same 12 captured checkpoints, same exposure as the
unconstrained reference (report 8n ``delayed_only``: 100 delayed episodes at
delay 30, saved RNG stream, frozen readout).  The delayed objective's
gradient is projected onto the null space of the state-Jacobian before the
optimiser step:

    J   = d s_flat / d theta   in R^(128 x P)

where s_flat is the response-step state on the same 16-instance probe batch
as section 8q (all 8 state coordinates = 128 constraints), theta is the
concatenated ``encoder_only`` parameter set (graph_encoder + pooling +
temporal_model state generation, excluding head and C/D), and

    u   = g - pinv(J) (J g)                (pinv rcond 1e-6)

u replaces the gradient; Adam then steps with full magnitude (alpha = 1, no
displacement budget, no scaling).  Recorded per step: raw gradient norm,
projection ratio |u|/|g|, and the *achieved* state displacement (measured,
never constrained).

The unconstrained baseline is 8n ``delayed_only`` (already measured; reused).

  python scripts/run_jacobian_projected.py [--shard 0 --num-shards 4] [--merge]
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

from artificial_system.environment.graph_environment import (  # noqa: E402
    GraphClassificationEnvironment,
    make_generic_graph,
)
from artificial_system.evaluation import state_content as sc  # noqa: E402
from artificial_system.evaluation.state_content import observe_instances  # noqa: E402
from artificial_system.experiments import coupling_rescue as cr  # noqa: E402
from artificial_system.models.gnn_ssm import build_model  # noqa: E402
from artificial_system.training.losses import classification_loss  # noqa: E402

STAGE_B_DIR = ROOT / "results" / "delay30_h7b"
PROBE_N = 16
PROBE_AT = sorted({1, 2, 3, 4, 5, *range(10, 101, 10)})
RCOND = 1e-6


def captured_checkpoints() -> list[tuple[str, Path]]:
    out = []
    for d in sorted(STAGE_B_DIR.glob("*__h7b_capture_alternate__t30")):
        p = d / "captured_checkpoint.pt"
        if p.exists():
            out.append((d.name.removesuffix("__h7b_capture_alternate__t30"), p))
    return out


@torch.no_grad()
def probe_displacement(model, probe_ds, s_cap, delay, marker_dim, generic_blank) -> float:
    was_training = model.training
    model.eval()
    st = sc.capture_trial_states(
        model, probe_ds, delay=delay, batch_size=PROBE_N,
        marker_dim=marker_dim, generic_blank=generic_blank,
    )
    if was_training:
        model.train()
    return float((st["pre_response"].detach().float() - s_cap).norm(dim=-1).mean())


def response_state_jacobian(model, probe_ds, delay, marker_dim, generic_blank,
                            params, batched: bool = False) -> torch.Tensor:
    """J = d s_flat / d theta in R^(128 x P) at the current parameters.

    batched=True uses is_grads_batched (vmap over the backward); it fails on
    this graph (in-place graph ops), so the default is the per-coordinate
    backward loop -- mathematically identical, just slower.
    """
    was_training = model.training
    model.eval()
    idxs = list(range(len(probe_ds)))
    state = model.initial_state(len(idxs))
    generic = observe_instances(probe_ds, idxs, keep_markers=False, marker_dim=marker_dim,
                                generic_blank=generic_blank)
    _, state, _ = model.step(generic, state)
    obs = observe_instances(probe_ds, idxs, True, marker_dim)
    _, state, _ = model.step(obs, state)
    for _ in range(delay):
        blank = observe_instances(probe_ds, idxs, keep_markers=False, marker_dim=marker_dim,
                                  generic_blank=generic_blank)
        _, state, _ = model.step(blank, state)
    s = state.reshape(-1)
    n = s.numel()
    if batched:
        # is_grads_batched returns one batched grad per input parameter:
        # g_k has shape [n_outputs, *params[k].shape]
        grads = torch.autograd.grad(
            s, params, grad_outputs=torch.eye(n), is_grads_batched=True,
        )
        J = torch.cat([g.reshape(g.shape[0], -1) for g in grads], dim=1)
    else:
        rows = []
        for j in range(n):
            g = torch.autograd.grad(s[j], params, retain_graph=(j < n - 1))
            rows.append(torch.cat([x.reshape(-1) for x in g]))
        J = torch.stack(rows)
    if was_training:
        model.train()
    return J.detach()


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

    from artificial_system.training.losses import IGNORE_INDEX
    ignore_index = IGNORE_INDEX

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
            losses.append(classification_loss(logits, targets, ignore_index))
            with torch.no_grad():
                actions = logits.argmax(-1)
            obs, _rewards, _done, _info = env.step(actions)

        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # gradient of the delayed objective (graph from the episode above)
        g_vec = torch.cat([p.grad.reshape(-1) for p in params]).detach()
        raw_norm = float(g_vec.norm())

        # J at the current parameters (requires grad mode; own forward graph)
        J = response_state_jacobian(
            model, probe_ds, delay, marker_dim, generic_blank, params
        ).detach()

        with torch.no_grad():
            Jg = J @ g_vec
            U = torch.linalg.pinv(J, rcond=RCOND)          # [P, 128]
            u_vec = g_vec - U @ Jg
            ratio = float(u_vec.norm() / g_vec.norm()) if raw_norm > 0 else float("nan")

            # write the projected gradient back into p.grad
            off = 0
            for p in params:
                n = p.numel()
                p.grad.copy_(u_vec[off:off + n].view_as(p))
                off += n
            optimizer.step()

        disp = probe_displacement(model, probe_ds, s_cap, delay, marker_dim, generic_blank)
        step_rows.append({
            "episode": ep, "raw_grad_norm": raw_norm, "projection_ratio": ratio,
            "projected_norm": raw_norm * ratio if raw_norm > 0 else 0.0,
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
    d_gnn = cr.param_group_deltas(snap_start, snap_end)

    summary = {
        "arm": "jacobian_projected",
        "seed": seed,
        "delay": int(delay),
        "source_checkpoint": str(ck_path.relative_to(ROOT)),
        "captured_from": ck_path.parent.name,
        "pre_analysis": pre_summary,
        "post_analysis": post_summary,
        "battery": battery,
        "episodes_run": int(episodes),
        "param_deltas": d_gnn,
        "rubric": rubric,
        "projection_ratio_mean": float(np.mean(ratios_f)) if ratios_f else float("nan"),
        "projection_ratio_min": float(np.min(ratios_f)) if ratios_f else float("nan"),
        "projection_ratio_max": float(np.max(ratios_f)) if ratios_f else float("nan"),
        "raw_grad_norm_mean": float(np.mean([r["raw_grad_norm"] for r in step_rows])),
        "displacement_mean": float(np.mean([r["displacement"] for r in step_rows])),
        "displacement_max": float(np.max([r["displacement"] for r in step_rows])),
        "jacobian_batched": True,
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
    ap.add_argument("--results-dir", default="results/delay30_jacobian")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--jacobian-loop", action="store_true",
                    help="compute J with a per-coordinate backward loop instead "
                         "of is_grads_batched (fallback if vmap misbehaves)")
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
        run_dir = out_dir / f"{tag}__jacobian_projected"
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
                  f"disp_max={s['displacement_max']:.3f} "
                  f"ALL={'PASS' if r['all_criteria'] else 'fail'}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"      ERROR in {tag}: {exc}", flush=True)
            raise
    _merge(out_dir)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
