#!/usr/bin/env python
"""Access-oscillation mechanism: phase-resolved measurements (report section 8u).

Re-runs the registered section 8q eps=0.2 budget recipe (identical line search,
RNG stream, frozen readout, 100 episodes) on the 9 disk-derived oscillator
checkpoints (amendment 1: seeds 0, 4, 6, 11, 12, 13, 16, 17, 21), instrumented:

  U1  per-episode: quick probe (delayed/reset/immediate), per-group gradient
      norms (post-clip), realised per-group step norms, Adam moment norms,
      probe-batch state capture before/after the step (per-step displacement
      norm + signed projection on the current readout direction w_r);
      dense state-content + geometry probes (episodes 1-30 every episode,
      31-60 every 2, 61-100 every 5): decode CV, substitution, controls,
      margins (actual response-step logits), margin distance, boundary
      position, dispersion, class separation;
      J instruments at fixed episodes {15, 45, 80} and at the first high->low
      and first low->high crossing (J at theta_pre; linear prediction vs
      realised per-step displacement; nulling residuals).
  U2  one interrupt per oscillator: at the first sustained high->low crossing
      (delayed < 0.6 after >= 2 episodes >= 0.9, e* >= 3): full battery, 5x
      repeated state-content evaluation (updates disabled), 10-episode
      frozen-hold control (data consumed, parameters and Adam state restored
      after each step), then exact resume (parameters + Adam state + env/torch
      RNG) to episode 100.

All evaluation-side instrumentation is wrapped in torch.random.fork_rng so it
cannot perturb the training RNG stream (amendment 3).  The re-run's per-episode
quick probes at section 8q's 15 probe episodes must reproduce the section 8q
records (checked inline, reported in the summary).

  python scripts/run_oscillation_mechanism.py [--shard 0 --num-shards 9] [--merge]
"""
from __future__ import annotations

import argparse
import copy
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
from artificial_system.experiments import coupling_rescue as cr  # noqa: E402
from artificial_system.models.gnn_ssm import build_model  # noqa: E402
from artificial_system.training.trainer import Trainer  # noqa: E402

import run_jacobian_projected as rjp  # noqa: E402
import run_state_budget as rsb  # noqa: E402

EPS = 0.2
EPISODES = 100
PROBE_N = rsb.PROBE_N
FIXED_J_EPS = {15, 45, 80}                     # amendment 2
DENSE_PROBES = (set(range(1, 31)) | set(range(32, 61, 2))
                | set(range(65, 101, 5)))
HOLD_EPISODES = 10
VAR_FLOOR = 0.01                               # 8o variance floor
OSC_SEEDS = {0, 4, 6, 11, 12, 13, 16, 17, 21}  # amendment 1


def osc_checkpoints() -> list[tuple[str, Path]]:
    out = []
    for tag, p in rsb.captured_checkpoints():
        seed = int(tag.split("seed")[1][:3])
        if seed in OSC_SEEDS:
            out.append((tag, p))
    return out


def group_map(model):
    """(name, [params]) for the tracked groups, trainable subset marked."""
    ssm = model.temporal_model
    groups = {
        "head": list(model.head.parameters()),
        "ssm_C": [ssm.C.weight],
        "ssm_D": [ssm.D.weight],
        "pooling": list(model.pooling.parameters()),
        "gnn": list(model.graph_encoder.parameters()),
    }
    if ssm.diagonal:
        groups["ssm_gen"] = [ssm.A_raw, ssm.B.weight]
    else:
        groups["ssm_gen"] = [ssm.B.weight]
    return groups


class InstrumentedStep:
    """Wraps optimizer.step (which is already the BudgetSearch interceptor).

    Logs per-group post-clip gradient norms (before the step) and realised
    per-group step norms + Adam moment norms (after the step).  Pure logging:
    the budget line search behaviour is untouched.
    """

    def __init__(self, model, optimizer, budget_search):
        self.model = model
        self.optimizer = optimizer
        self.budget = budget_search
        self.groups = group_map(model)
        self.log: list[dict] = []
        self.wrapped = optimizer.step
        optimizer.step = self.step  # type: ignore[method-assign]

    def _group_flat(self, name: str) -> torch.Tensor:
        return torch.cat([p.detach().flatten() for p in self.groups[name]])

    def _grad_norms(self) -> dict:
        out = {}
        for name, ps in self.groups.items():
            tot = 0.0
            for p in ps:
                if p.requires_grad and p.grad is not None:
                    tot += float(p.grad.detach().norm()) ** 2
            out[f"grad_norm_{name}"] = tot ** 0.5
        return out

    def _moment_norms(self) -> dict:
        out = {}
        for name, ps in self.groups.items():
            ea = es = 0.0
            for p in ps:
                st = self.optimizer.state.get(p)
                if st and "exp_avg" in st:
                    ea += float(st["exp_avg"].norm()) ** 2
                    es += float(st["exp_avg_sq"].norm()) ** 2
            out[f"moment_norm_{name}"] = ea ** 0.5
            out[f"moment2_norm_{name}"] = es ** 0.5
        return out

    def step(self, closure=None):
        pre = {name: self._group_flat(name) for name in self.groups}
        grads = self._grad_norms()
        out = self.wrapped(closure)
        step_norms = {}
        for name in self.groups:
            step_norms[f"step_norm_{name}"] = float(
                (self._group_flat(name) - pre[name]).norm())
        self.log.append({"step": self.budget.step_idx - 1, **grads,
                         **step_norms, **self._moment_norms()})
        return out


def dense_probe(model, test_ds, blank, delay, marker_dim, generic_blank, seed,
                run_dir, tag) -> dict:
    """State-content (run_state_content) + geometry at one phase."""
    with torch.random.fork_rng():
        summary = cr._state_summary(model, test_ds, delay, marker_dim,
                                    generic_blank, seed, run_dir, tag)
        model.eval()
        st = sc.capture_trial_states(model, test_ds, delay=delay, batch_size=50,
                                     marker_dim=marker_dim,
                                     generic_blank=generic_blank)
        model.eval()
        X = st["pre_response"].detach().float()
        labels = st["_label"].numpy().astype(int)
        margin = st["_signed_margin"].numpy().astype(np.float64)
        bnd = rg.boundary_statistics(margin, labels)
        w_r = rg.readout_direction(model, blank, X.mean(0)).detach().double()
        w_rn = w_r / w_r.norm() if float(w_r.norm()) > 0 else w_r
        w_p, mu = rg.probe_direction(st["pre_response"], st["_label"], lam=1.0)
        var = []
        Xn = X.numpy()
        for c in (0, 1):
            Xc = Xn[labels == c]
            if len(Xc) > 1:
                var.append(Xc.var(0, ddof=1))
        pooled = np.maximum(np.mean(var, axis=0), VAR_FLOOR) if var else np.full(X.shape[1], VAR_FLOOR)
        row = {
            "abs_margin_mean": float(np.abs(margin).mean()),
            "m_dist": float(np.abs(margin).mean() / float(w_r.norm())),
            "state_norm_mean": float(X.norm(dim=-1).mean()),
            "disp_state": float(np.sqrt(pooled.mean())),
            "class_sep_state": float(np.linalg.norm(Xn[labels == 1].mean(0) - Xn[labels == 0].mean(0))),
            "readout_norm": float(w_r.norm()),
            **{k: float(v) for k, v in bnd.items()},
        }
    row.update({k: summary.get(k) for k in (
        "decode_pre_response", "decode_baseline", "subst_diff_follows",
        "subst_same_answer_accuracy", "subst_zero_accuracy",
        "subst_shuffled_accuracy")})
    # U3 sidecar vectors: ridge decode direction and class-mean difference
    wp_np = np.asarray(w_p, dtype=np.float64).ravel()
    wpl = float(np.linalg.norm(wp_np))
    if wpl > 0:
        wp_np = wp_np / wpl
    row["_wp"] = wp_np
    row["_dmu"] = Xn[labels == 1].mean(0).astype(np.float64) \
        - Xn[labels == 0].mean(0).astype(np.float64)
    return row


def probe_batch_states(model, probe_ds, delay, marker_dim, generic_blank) -> torch.Tensor:
    with torch.random.fork_rng():
        was_training = model.training
        model.eval()
        st = sc.capture_trial_states(model, probe_ds, delay=delay,
                                     batch_size=PROBE_N, marker_dim=marker_dim,
                                     generic_blank=generic_blank)
        if was_training:
            model.train()
        return st["pre_response"].detach().float().clone()


def jacobian_at(model, enc, theta_target, probe_ds, delay, marker_dim,
                generic_blank) -> torch.Tensor:
    """J at theta_target; restores whatever params were live when called."""
    live = [p.detach().clone() for p in enc]
    with torch.random.fork_rng():
        with torch.no_grad():
            for p, t in zip(enc, theta_target):
                p.copy_(t)
            J = rjp.response_state_jacobian(model, probe_ds, delay, marker_dim,
                                            generic_blank, enc)
        with torch.no_grad():
            for p, t in zip(enc, live):
                p.copy_(t)
    return J


def j_row(kind: str, ep: int, J: torch.Tensor, d_theta: torch.Tensor,
          ds_norm: float) -> dict:
    lin = J @ d_theta
    lin_norm = float(lin.norm())
    U = torch.linalg.pinv(J, rcond=1e-6)
    d_safe = d_theta - U @ lin
    return {
        "kind": kind, "episode": ep,
        "d_theta_norm": float(d_theta.norm()),
        "lin_pred": lin_norm,
        "ds_norm": ds_norm,
        "curvature_ratio": (ds_norm / lin_norm) if lin_norm > 0 else float("nan"),
        "null_frac": (float((d_theta - d_safe).norm()) / float(d_theta.norm())
                      if float(d_theta.norm()) > 0 else float("nan")),
        "residual_lin": (float((J @ d_safe).norm()) / lin_norm
                         if lin_norm > 0 else float("nan")),
    }


class FreezeUpdates:
    """U2 frozen-hold: steps run (data stream consumed) but parameters AND
    Adam state are restored after every step.

    Calls the *pristine* Adam step (captured before BudgetSearch wrapped
    optimizer.step) so the budget line search's cumulative tracker and logs,
    and the instrumentation, are untouched; everything is rolled back anyway
    before the resume.
    """

    def __init__(self, model, optimizer, pristine_step):
        self.model = model
        self.optimizer = optimizer
        self.pristine = pristine_step
        self.trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer.step = self.step  # type: ignore[method-assign]

    def step(self, closure=None):
        theta = [p.detach().clone() for p in self.trainable]
        snap = copy.deepcopy(self.optimizer.state_dict())
        out = self.pristine(closure)
        with torch.no_grad():
            for p, t in zip(self.trainable, theta):
                p.copy_(t)
        self.optimizer.load_state_dict(snap)
        return out

    def remove(self, restore_step):
        # re-install the instrumentation chain (instrumented step, which wraps
        # the budget search)
        self.optimizer.step = restore_step  # type: ignore[method-assign]


def run_one(tag: str, ck_path: Path, run_dir: Path) -> dict:
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
    enc = [p for p in model.parameters() if p.requires_grad]

    generic_blank = None
    if task_cfg.get("blank_mode", "no_markers") == "generic":
        generic_blank = make_generic_graph(
            int(task_cfg.get("num_nodes", 100)),
            int(cfg.get("graph", {}).get("node_feature_dim", 2)),
            seed * 2000 + 2,
        )

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
        model, test_ds, delay, marker_dim, generic_blank, seed, run_dir, "pre")
    snap_start = cr.param_group_snapshot(model)

    with torch.random.fork_rng():
        model.eval()
        st_cap = sc.capture_trial_states(model, probe_ds, delay=delay,
                                         batch_size=PROBE_N, marker_dim=marker_dim,
                                         generic_blank=generic_blank)
        s_cap = st_cap["pre_response"].detach().float()
        if generic_blank is not None:
            st_cap = sc.capture_trial_states(model, probe_ds, delay=delay,
                                             batch_size=PROBE_N, marker_dim=marker_dim,
                                             generic_blank=generic_blank)
            s_cap = st_cap["pre_response"].detach().float()

    blank = sc.observe_instances(test_ds, [0], keep_markers=False,
                                 marker_dim=marker_dim, generic_blank=generic_blank)

    budget_search = rsb.BudgetSearch(model, optimizer, EPS, s_cap, probe_ds,
                                     delay, marker_dim, generic_blank, fast=False)
    instrument = InstrumentedStep(model, optimizer, budget_search)
    trainer = Trainer(model, env, optimizer, train_cfg)

    ep_rows: list[dict] = []
    probe_rows: list[dict] = []
    j_rows: list[dict] = []
    wp_list: list[np.ndarray] = []
    dmu_list: list[np.ndarray] = []
    ep_wp: list[int] = []
    interrupt_info: dict | None = None
    ring: dict | None = None          # 1-step ring: theta_pre + s_pre + ds_norm
    trans_done = {"hl": False, "lh": False}
    prev1 = prev2 = None

    def quick() -> dict:
        with torch.random.fork_rng():
            return cr.quick_probe(model, test_ds, delay, marker_dim, generic_blank)

    q0 = quick()
    ep_rows.append({"episode": 0, "frozen": False, **q0})
    print(f"  pre: delayed={q0['delayed_intact']:.3f}", flush=True)

    for ep in range(1, EPISODES + 1):
        # ---- ring snapshot at theta_pre ----
        with torch.no_grad():
            theta_pre = [p.detach().clone() for p in enc]
        s_pre = probe_batch_states(model, probe_ds, delay, marker_dim, generic_blank)

        # fixed-episode J instrument (params are at theta_pre right now)
        J_fix = None
        if ep in FIXED_J_EPS:
            with torch.random.fork_rng():
                J_fix = rjp.response_state_jacobian(model, probe_ds, delay,
                                                    marker_dim, generic_blank, enc)

        trainer.train(1)

        s_post = probe_batch_states(model, probe_ds, delay, marker_dim, generic_blank)
        with torch.random.fork_rng():
            model.eval()
            w_r = rg.readout_direction(model, blank, s_post.mean(0)).detach().double()
        w_rn = w_r / w_r.norm() if float(w_r.norm()) > 0 else w_r
        ds = s_post - s_pre
        ds_norm = float(ds.norm(dim=-1).mean())
        ds_proj = float((ds.double() @ w_rn).mean())

        q = quick()
        row = {"episode": ep, "frozen": False, **q,
               "ds_norm": ds_norm, "ds_proj_wr": ds_proj,
               "raw_step_norm": float(budget_search.log[-1]["raw_step_norm"]),
               "alpha": float(budget_search.log[-1]["alpha"]),
               "accepted_norm": float(budget_search.log[-1]["accepted_norm"]),
               "cum_displacement": float(budget_search.log[-1]["displacement"])}
        if instrument.log:
            row.update(instrument.log[-1])
        ep_rows.append(row)

        # ---- fixed-episode J row ----
        if J_fix is not None:
            with torch.no_grad():
                d_theta = torch.cat([(p.detach() - t).reshape(-1)
                                     for p, t in zip(enc, theta_pre)])
            j_rows.append(j_row("fixed", ep, J_fix, d_theta, ds_norm))
            del J_fix

        # ---- transition J instruments (first of each kind) ----
        if prev1 is not None:
            kind = None
            if prev1 >= 0.9 and q["delayed_intact"] < 0.6:
                kind = "hl_crossing"
            elif prev1 < 0.6 and q["delayed_intact"] >= 0.9:
                kind = "lh_crossing"
            if kind and not trans_done[kind]:
                trans_done[kind] = True
                with torch.no_grad():
                    d_theta = torch.cat([(p.detach() - t).reshape(-1)
                                         for p, t in zip(enc, theta_pre)])
                J = jacobian_at(model, enc, theta_pre, probe_ds, delay,
                                marker_dim, generic_blank)
                j_rows.append(j_row(kind, ep, J, d_theta, ds_norm))
                del J
                print(f"  {kind} at ep {ep}: ds={ds_norm:.4f}", flush=True)

        # ---- U2 interrupt: first sustained high->low crossing ----
        if (interrupt_info is None and ep >= 3 and prev2 is not None
                and prev2 >= 0.9 and prev1 >= 0.9 and q["delayed_intact"] < 0.6):
            interrupt_info = interrupt_branch(
                model, enc, optimizer, env, trainer, test_ds, probe_ds, blank,
                delay, marker_dim, generic_blank, seed, run_dir, ep, instrument)

        # ---- dense probes (post-resume params = post-e_star state: valid) ----
        if ep in DENSE_PROBES:
            prow = dense_probe(model, test_ds, blank, delay, marker_dim,
                               generic_blank, seed, run_dir, f"d{ep:03d}")
            wp_list.append(prow.pop("_wp"))
            dmu_list.append(prow.pop("_dmu"))
            ep_wp.append(ep)
            probe_rows.append({"episode": ep, **prow})

        prev2, prev1 = prev1, q["delayed_intact"]
        ring = {"theta": theta_pre, "s_pre": s_pre, "ds_norm": ds_norm}

        if ep % 10 == 0:
            pd.DataFrame(ep_rows).to_csv(run_dir / "per_episode.csv", index=False)
            pd.DataFrame(probe_rows).to_csv(run_dir / "dense_probes.csv", index=False)
            print(f"  ep {ep}: delayed={q['delayed_intact']:.3f} "
                  f"cum_disp={row['cum_displacement']:.4f}", flush=True)

    post_summary = cr._state_summary(
        model, test_ds, delay, marker_dim, generic_blank, seed, run_dir, "post")
    battery = cr._final_battery(model, test_ds, delay, marker_dim, generic_blank, seed)
    snap_end = cr.param_group_snapshot(model)
    rubric = cr.success_rubric(pre_summary, post_summary, battery)

    pd.DataFrame(ep_rows).to_csv(run_dir / "per_episode.csv", index=False)
    pd.DataFrame(probe_rows).to_csv(run_dir / "dense_probes.csv", index=False)
    pd.DataFrame(j_rows).to_csv(run_dir / "j_instruments.csv", index=False)
    if wp_list:
        np.savez(run_dir / "dense_dirs.npz",
                 episode=np.asarray(ep_wp, dtype=int),
                 wp=np.stack(wp_list), dmu=np.stack(dmu_list))
    search_log = pd.DataFrame(budget_search.log)
    search_log.to_csv(run_dir / "step_log.csv", index=False)

    # ---- determinism check vs section 8q records ----
    q_path = ROOT / "results" / "delay30_maintenance" / f"{tag}__budget_eps0.2" / "training_probes.csv"
    q_check = {"checked": False}
    if q_path.exists():
        q8 = pd.read_csv(q_path)
        mine = pd.DataFrame(ep_rows)
        diffs = []
        for _, r in q8.iterrows():
            m = mine[mine.episode == int(r["episodes_done"])]
            if len(m) == 0:
                continue
            for k in ("delayed_intact", "immediate", "delayed_reset"):
                diffs.append(abs(float(m.iloc[0][k]) - float(r[k])))
        q_check = {"checked": True, "n_compared": len(diffs),
                   "max_abs_diff": max(diffs) if diffs else 0.0,
                   "exact": bool(diffs and max(diffs) == 0.0)}
        print(f"  8q determinism: max|d|={q_check['max_abs_diff']:.2e} "
              f"({q_check['n_compared']} values)", flush=True)

    summary = {
        "arm": "oscillation_mechanism",
        "budget": float(EPS),
        "seed": seed,
        "delay": int(delay),
        "source_checkpoint": str(ck_path.relative_to(ROOT)),
        "pre_analysis": pre_summary,
        "post_analysis": post_summary,
        "battery": battery,
        "episodes_run": int(EPISODES),
        "param_deltas": cr.param_group_deltas(snap_start, snap_end),
        "rubric": rubric,
        "steps": int(budget_search.step_idx),
        "interrupt": interrupt_info,
        "transitions_found": {k: v for k, v in trans_done.items()},
        "q_determinism_check": q_check,
    }
    with (run_dir / "arm_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary


def interrupt_branch(model, enc, optimizer, env, trainer, test_ds, probe_ds,
                     blank, delay, marker_dim, generic_blank, seed, run_dir,
                     e_star, instrument) -> dict:
    """U2: battery, 5x no-update repeats, frozen-hold, exact resume."""
    print(f"  INTERRUPT at ep {e_star} (sustained high->low)", flush=True)
    params_snap = [p.detach().clone() for p in enc]
    opt_snap = copy.deepcopy(optimizer.state_dict())
    torch_snap = torch.get_rng_state()
    env_snap = env.rng.get_state()

    # (a) full battery at the low phase
    with torch.random.fork_rng():
        battery = cr._final_battery(model, test_ds, delay, marker_dim,
                                    generic_blank, seed)

    # (b) 5x repeated state-content evaluation, updates disabled
    repeats = []
    for i in range(5):
        s = cr._state_summary(model, test_ds, delay, marker_dim, generic_blank,
                              seed, run_dir, f"interrupt{i}")
        repeats.append({k: s.get(k) for k in (
            "decode_pre_response", "subst_diff_follows",
            "subst_same_answer_accuracy")})

    # (c) frozen-hold: 10 episodes, data consumed, parameters + Adam restored;
    # bypasses the budget search and instrumentation entirely (pristine Adam)
    freeze = FreezeUpdates(model, optimizer, budget_search_pristine(instrument))
    hold_rows = []
    for k in range(1, HOLD_EPISODES + 1):
        trainer.train(1)
        with torch.random.fork_rng():
            q = cr.quick_probe(model, test_ds, delay, marker_dim, generic_blank)
        hold_rows.append({"hold_episode": k, **q})
        print(f"    hold {k}: delayed={q['delayed_intact']:.3f}", flush=True)
    freeze.remove(instrument.step)

    # (d) exact resume
    with torch.no_grad():
        for p, t in zip(enc, params_snap):
            p.copy_(t)
    optimizer.load_state_dict(opt_snap)
    torch.set_rng_state(torch_snap)
    env.rng.set_state(env_snap)
    restored = all(bool(torch.equal(p.detach(), t))
                   for p, t in zip(enc, params_snap))
    info = {
        "e_star": int(e_star),
        "battery_delayed": float(battery["delayed_intact"]),
        "battery": {k: float(v) for k, v in battery.items()
                    if isinstance(v, (int, float))},
        "repeats": repeats,
        "frozen_hold": hold_rows,
        "params_restored_bitexact": bool(restored),
    }
    print(f"  resume: params_restored={restored}", flush=True)
    return info


def budget_search_pristine(instrument) -> object:
    """The pristine Adam step bound method captured before BudgetSearch wrapped
    optimizer.step."""
    return instrument.budget.orig_step


def _merge(out_dir: Path) -> None:
    merged: dict = {}
    for f in sorted(out_dir.glob("*__*/arm_summary.json")):
        merged[f.parent.name] = json.loads(f.read_text())
    (out_dir / "summary.json").write_text(json.dumps(merged, indent=2, default=float))
    print(f"merged {len(merged)} runs -> {out_dir / 'summary.json'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/delay30_oscillation")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=9)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge:
        _merge(out_dir)
        return 0

    ckpts = osc_checkpoints()
    mine = [c for i, c in enumerate(ckpts) if i % args.num_shards == args.shard]
    print(f"[shard {args.shard}/{args.num_shards}] {len(mine)} of {len(ckpts)} "
          f"oscillators", flush=True)
    for tag, ck_path in mine:
        run_dir = out_dir / f"{tag}__osc"
        if (run_dir / "arm_summary.json").exists():
            print(f"[skip] {tag} (complete)", flush=True)
            continue
        print(f"[run ] {tag} -> {run_dir}", flush=True)
        s = run_one(tag, ck_path, run_dir)
        print(f"      rubric delayed_ok={s['rubric']['delayed_ok']} "
              f"transitions={s['transitions_found']}", flush=True)
    _merge(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
