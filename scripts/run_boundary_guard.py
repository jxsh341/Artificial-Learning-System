#!/usr/bin/env python
"""V3 guard arm: bvm-guarded budget line search (report section 8v).

Identical registered recipe to the V1/V2 runs (section 8q eps=0.2 full-ladder
line search, frozen readout, 100 episodes, same seeds), with one additional
acceptance criterion at every ladder candidate, per registration + amendment 2:

  reject candidate alpha if |bvm_pre| < 0.5 and |bvm_post| >= 0.5, in either
  direction (the cloud would pass beyond either class edge); steps that move
  |bvm| toward 0 (toward the boundary-aligned region) are never restricted.

bvm is read from the line-search's own replay states (probe batch, registered
instrument); alpha = 0 (rejection) remains admissible.  Per-episode online
probes (the V1/V2 instrument) record the test-set bvm trajectory for scoring;
the guard never touches that stream.

  python scripts/run_boundary_guard.py [--shard 0 --num-shards 12] [--merge]
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

import run_online_fragility as rf  # noqa: E402
import run_state_budget as rsb  # noqa: E402
from artificial_system.evaluation import readout_geometry as rg  # noqa: E402
from artificial_system.evaluation import state_content as sc  # noqa: E402

ALPHA_LADDER = rsb.ALPHA_LADDER
T_OFF = 0.5       # registered (amendment 2): wrong-side threshold, either sign
EPS = 0.2

BASE = ROOT / "results" / "delay30_fragility_guard"


class GuardedBudgetSearch(rsb.BudgetSearch):
    """BudgetSearch + registered V3 guard (amendment-3 instrument).

    Post-candidate bvm is PREDICTED ON THE TEST CLOUD via the readout's exact
    linearity in the state:  bvm_pred = bvm_test_pre - (mid_cand - mid_pre) /
    sep_test_pre.  bvm_test_pre / sep_test_pre come from the previous
    episode's online probe (the pre-step test-cloud state; injected per
    episode by the runner); mid_cand / mid_pre are the class-mid margins of
    the candidate / pre-step replay states (already computed per candidate).
    Guard predicate (registered): reject a candidate iff |bvm_test_pre| < 0.5
    and |bvm_pred| >= 0.5 (either direction); steps toward 0 unrestricted;
    alpha = 0 fallback.  If |sep_test_pre| < 0.05 (never observed) the step is
    logged unguarded.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_states = None      # cached candidate states (disp_for)
        self._guard_rejects = 0
        self._guard_max_streak = 0
        self._guard_streak = 0
        self._unguarded_steps = 0
        self.test_bvm_pre = float("nan")   # injected per episode by runner
        self.test_sep_pre = float("nan")

    @torch.no_grad()
    def _displacement_full(self):
        was_training = self.model.training
        self.model.eval()
        st = sc.capture_trial_states(
            self.model, self.probe_ds, delay=self.delay, batch_size=PROBE_B,
            marker_dim=self.marker_dim, generic_blank=self.generic_blank,
        )
        if was_training:
            self.model.train()
        return st

    @staticmethod
    @torch.no_grad()
    def _mid_of(st) -> float:
        margin = st["_signed_margin"].numpy().astype(np.float64)
        labels = st["_label"].numpy().astype(int)
        bnd = rg.boundary_statistics(margin, labels)
        return 0.5 * (bnd["proj_readout_class0"] + bnd["proj_readout_class1"])

    @torch.no_grad()
    def _disp_for(self, alpha, theta_pre, deltas):
        for p, t, d in zip(self.trainable, theta_pre, deltas):
            p.copy_(t + alpha * d)
        st = self._displacement_full()
        self._last_states = st
        disp = float((st["pre_response"].detach().float() - self.s_cap)
                     .norm(dim=-1).mean())
        return disp, alpha

    @torch.no_grad()
    def step(self, closure=None):
        theta_pre = [p.detach().clone() for p in self.trainable]
        out = self.orig_step(closure)
        deltas = [p.detach() - t for p, t in zip(self.trainable, theta_pre)]

        mid_pre = self._mid_of(self._displacement_full())
        on_side = abs(self.test_bvm_pre) < T_OFF
        sep_ok = abs(self.test_sep_pre) >= 0.05
        guard_active = bool(on_side and sep_ok and self.test_bvm_pre == self.test_bvm_pre)
        if not sep_ok:
            self._unguarded_steps += 1

        accepted = None
        guard_rej_this = 0
        bvm_pred_used = float("nan")
        self._last_states = None
        for alpha in ALPHA_LADDER:
            d, _ = self._disp_for(alpha, theta_pre, deltas)
            if guard_active:
                mid_cand = self._mid_of(self._last_states)
                bvm_pred = (self.test_bvm_pre
                            - (mid_cand - mid_pre) / self.test_sep_pre)
                bvm_pred_used = bvm_pred
                guard_ok = abs(bvm_pred) < T_OFF
            else:
                guard_ok = True
            if d <= self.eps and guard_ok:
                accepted = (alpha, d)
                break
            if d <= self.eps and not guard_ok:
                guard_rej_this += 1
        if accepted is None:  # alpha = 0 always admissible
            for p, t in zip(self.trainable, theta_pre):
                p.copy_(t)
            accepted = (0.0, float("nan"))
        alpha, disp = accepted
        if alpha == 0.0:
            for p, t in zip(self.trainable, theta_pre):
                p.copy_(t)

        self._guard_rejects += guard_rej_this
        self._guard_streak = self._guard_streak + 1 if guard_rej_this else 0
        self._guard_max_streak = max(self._guard_max_streak, self._guard_streak)

        raw_norm = float(torch.sqrt(sum(d.norm() ** 2 for d in deltas)))
        self.log.append({
            "step": self.step_idx, "alpha": alpha,
            "raw_step_norm": raw_norm,
            "accepted_norm": alpha * raw_norm,
            "displacement": disp, "eps": self.eps,
            "bvm_test_pre": self.test_bvm_pre,
            "bvm_pred": bvm_pred_used,
            "guard_rejects": guard_rej_this,
            "guard": True,
        })
        self.step_idx += 1
        return out


PROBE_B = rf.PROBE_N  # line-search replay batch = the registered probe batch


def run_one(tag: str, ck_path: Path, run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    import torch as _torch
    ck = _torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    seed = int(ck["seed"])
    delay = int(ck.get("delay", 30))

    task_cfg = dict(cfg.get("task", {}))
    task_cfg["delay"] = delay
    marker_dim = int(task_cfg.get("marker_dim", 2))
    train_cfg = dict(cfg.get("training", {}))

    import run_oscillation_mechanism as rom
    datasets = rf.cr.build_datasets_from_cfg(cfg)
    test_ds = datasets["test"]
    from artificial_system.data.graph_tasks import ConnectivityDataset
    probe_ds = ConnectivityDataset(
        graphs=list(test_ds.graphs[:rf.PROBE_N]),
        labels=test_ds.labels[:rf.PROBE_N].clone(),
    )

    model = rf.build_model(
        cfg, node_feature_dim=int(cfg.get("graph", {}).get("node_feature_dim", 2)),
        edge_feature_dim=0, num_actions=2, num_nodes=None,
    )
    model.load_state_dict(ck["model_state"])
    del ck["model_state"]
    rf.cr.set_trainables(model, "encoder_only")

    generic_blank = None
    if task_cfg.get("blank_mode", "no_markers") == "generic":
        generic_blank = rf.make_generic_graph(
            int(task_cfg.get("num_nodes", 100)),
            int(cfg.get("graph", {}).get("node_feature_dim", 2)),
            seed * 2000 + 2,
        )

    env = rf.GraphClassificationEnvironment(datasets["train"], task_cfg,
                                           seed=seed * 2000 + 2)
    rng_states = ck.get("rng_states") or {}
    if "env_del" in rng_states:
        env.rng.set_state(rng_states["env_del"])

    optimizer = _torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    pre_summary = rf.cr._state_summary(
        model, test_ds, delay, marker_dim, generic_blank, seed, run_dir, "pre")
    snap_start = rf.cr.param_group_snapshot(model)

    blank = rf.sc.observe_instances(test_ds, [0], keep_markers=False,
                                    marker_dim=marker_dim,
                                    generic_blank=generic_blank)

    with _torch.random.fork_rng():
        model.eval()
        st_cap = rf.sc.capture_trial_states(model, probe_ds, delay=delay,
                                            batch_size=rf.PROBE_N,
                                            marker_dim=marker_dim,
                                            generic_blank=generic_blank)
        s_cap = st_cap["pre_response"].detach().float()
        if generic_blank is not None:
            st_cap = rf.sc.capture_trial_states(model, probe_ds, delay=delay,
                                                batch_size=rf.PROBE_N,
                                                marker_dim=marker_dim,
                                                generic_blank=generic_blank)
            s_cap = st_cap["pre_response"].detach().float()

    budget_search = GuardedBudgetSearch(model, optimizer, EPS, s_cap, probe_ds,
                                        delay, marker_dim, generic_blank,
                                        fast=False)
    instrument = rom.InstrumentedStep(model, optimizer, budget_search)
    trainer = rf.Trainer(model, env, optimizer, train_cfg)

    ep_rows: list[dict] = []
    prev_bvm = None

    def quick():
        with _torch.random.fork_rng():
            return rf.cr.quick_probe(model, test_ds, delay, marker_dim,
                                     generic_blank)

    q0 = quick()
    p0 = rf.online_probe(model, test_ds, delay, marker_dim, generic_blank)
    prev_bvm = p0["boundary_vs_mid"]
    # amendment-3 instrument: the guard's pre-step test-cloud state is the
    # previous episode's online-probe values (no param change in between)
    budget_search.test_bvm_pre = prev_bvm
    budget_search.test_sep_pre = p0["class_margin_separation"]
    ep_rows.append({"episode": 0, "frozen": False, **q0, **p0,
                    "bvm_velocity": np.nan, "bvm_test_pre": prev_bvm,
                    "guard_rejects": 0})
    print(f"  pre: delayed={q0['delayed_intact']:.3f} bvm={prev_bvm:+.4f}",
          flush=True)

    for ep in range(1, 101):
        s_pre = rom.probe_batch_states(model, probe_ds, delay, marker_dim,
                                       generic_blank)
        trainer.train(1)
        s_post = rom.probe_batch_states(model, probe_ds, delay, marker_dim,
                                        generic_blank)
        with _torch.random.fork_rng():
            model.eval()
            w_r = rg.readout_direction(model, blank, s_post.mean(0)).detach().double()
        w_rn = w_r / w_r.norm() if float(w_r.norm()) > 0 else w_r
        ds_norm = float((s_post - s_pre).norm(dim=-1).mean())

        q = quick()
        prow = rf.online_probe(model, test_ds, delay, marker_dim, generic_blank)
        row = {"episode": ep, "frozen": False, **q,
               "ds_norm": ds_norm,
               "raw_step_norm": float(budget_search.log[-1]["raw_step_norm"]),
               "alpha": float(budget_search.log[-1]["alpha"]),
               "accepted_norm": float(budget_search.log[-1]["accepted_norm"]),
               "cum_displacement": float(budget_search.log[-1]["displacement"]),
               "guard_rejects": int(budget_search.log[-1]["guard_rejects"]),
               "bvm_test_pre": (float(budget_search.log[-1]["bvm_test_pre"])
                                if "bvm_test_pre" in budget_search.log[-1]
                                else float("nan")),
               **prow}
        if instrument.log:
            row.update(instrument.log[-1])
        bvm = prow["boundary_vs_mid"]
        row["bvm_velocity"] = (bvm - prev_bvm
                               if (bvm == bvm and prev_bvm == prev_bvm)
                               else float("nan"))
        prev_bvm = bvm
        # next episode's guard pre-state = this episode's test-cloud probe
        budget_search.test_bvm_pre = bvm
        budget_search.test_sep_pre = float(prow["class_margin_separation"])
        ep_rows.append(row)

        if ep % 10 == 0:
            pd.DataFrame(ep_rows).to_csv(run_dir / "per_episode.csv", index=False)
            print(f"  ep {ep}: delayed={q['delayed_intact']:.3f} bvm={bvm:+.4f} "
                  f"a={row['alpha']:.4f} rej={row['guard_rejects']} "
                  f"cum={row['cum_displacement']:.4f}", flush=True)

    post_summary = rf.cr._state_summary(
        model, test_ds, delay, marker_dim, generic_blank, seed, run_dir, "post")
    battery = rf.cr._final_battery(model, test_ds, delay, marker_dim,
                                   generic_blank, seed)
    rubric = rf.cr.success_rubric(pre_summary, post_summary, battery)

    pd.DataFrame(ep_rows).to_csv(run_dir / "per_episode.csv", index=False)
    log = pd.DataFrame(budget_search.log)
    log.to_csv(run_dir / "step_log.csv", index=False)

    return {
        "tag": tag, "seed": seed, "delay": delay,
        "rubric": rubric,
        "param_deltas": rf.cr.param_group_deltas(snap_start,
                                                 rf.cr.param_group_snapshot(model)),
        "guard_stats": {
            "instrument": "predicted_test_bvm (amendment 3)",
            "total_rejects": int(budget_search._guard_rejects),
            "max_consecutive_reject_episodes": int(budget_search._guard_max_streak),
            "unguarded_steps": int(budget_search._unguarded_steps),
            "rejected_steps_frac": (float((log["alpha"] == 0.0).mean())
                                    if len(log) else None),
        },
        "n_episodes": 100,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    BASE.mkdir(parents=True, exist_ok=True)
    (BASE / "logs").mkdir(exist_ok=True)

    cohort = rf.cohort_checkpoints()
    jobs = [(t, p) for i, (t, p) in enumerate(cohort)
            if i % args.num_shards == args.shard]
    print(f"shard {args.shard}/{args.num_shards}: "
          f"{[int(t.split('seed')[1][:3]) for t, _ in jobs]}", flush=True)
    for tag, p in jobs:
        seed = int(tag.split("seed")[1][:3])
        run_dir = BASE / f"D800_interleaved_seed{seed:03d}__guard"
        if (run_dir / "arm_summary.json").exists():
            print(f"skip {tag} (already done)", flush=True)
            continue
        print(f"=== {tag} ===", flush=True)
        s = run_one(tag, p, run_dir)
        (run_dir / "arm_summary.json").write_text(json.dumps(s, indent=2,
                                                             default=str))
        print(f"  saved {run_dir / 'arm_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
