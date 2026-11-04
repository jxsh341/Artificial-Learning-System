#!/usr/bin/env python
"""Online fragility indicator: light per-episode re-runs (report section 8v).

V1/V2 runner (V3 has its own guarded arm in run_boundary_guard.py).  Re-runs
the registered section 8q eps=0.2 budget recipe (identical line search, RNG
stream, frozen readout, 100 episodes, no interrupt branch, no dense probes, no
J instruments) on all 12 captured delay-30 checkpoints -- 9 oscillators (0, 4,
6, 11, 12, 13, 16, 17, 21), seed 14 (permanent collapse comparator), seeds
18/22 (stable comparators).

Added per-episode online instrumentation (registered): the section 8u
boundary-position instrument verbatim -- boundary_vs_mid from
rg.boundary_statistics (readout margins and class labels on the captured
state cloud), plus the one-episode velocity v = bvm_e - bvm_{e-1}.  Wrapped in
torch.random.fork_rng so it cannot perturb the training stream.  Rule inputs
contain no behavioural accuracy quantities (V2 fairness criterion, registered).

Integrity check: per-episode quick probes must reproduce the section 8q
records (checked inline, reported in the summary).

  python scripts/run_online_fragility.py [--shard 0 --num-shards 12] [--merge]
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
from artificial_system.experiments import coupling_rescue as cr  # noqa: E402
from artificial_system.models.gnn_ssm import build_model  # noqa: E402
from artificial_system.training.trainer import Trainer  # noqa: E402

import run_oscillation_mechanism as rom  # noqa: E402  (InstrumentedStep, probe helpers)
import run_state_budget as rsb  # noqa: E402

EPS = 0.2
EPISODES = 100
PROBE_N = rsb.PROBE_N

BASE = ROOT / "results" / "delay30_fragility"


def cohort_checkpoints() -> list[tuple[str, Path]]:
    """All 12 captured delay-30 checkpoints (registered 8v cohort)."""
    return sorted(rsb.captured_checkpoints(), key=lambda tp: int(tp[0].split("seed")[1][:3]))


def online_probe(model, test_ds, delay, marker_dim, generic_blank) -> dict:
    """The section 8u boundary-position instrument, applied now (light:
    state capture + readout margins + geometry only; no decode CV, no
    substitution battery)."""
    with torch.random.fork_rng():
        was_training = model.training
        model.eval()
        st = sc.capture_trial_states(model, test_ds, delay=delay, batch_size=50,
                                     marker_dim=marker_dim,
                                     generic_blank=generic_blank)
        if was_training:
            model.train()
        X = st["pre_response"].detach().float()
        labels = st["_label"].numpy().astype(int)
        margin = st["_signed_margin"].numpy().astype(np.float64)
        bnd = rg.boundary_statistics(margin, labels)
        blank = sc.observe_instances(test_ds, [0], keep_markers=False,
                                     marker_dim=marker_dim,
                                     generic_blank=generic_blank)
        w_r = rg.readout_direction(model, blank, X.mean(0)).detach().double()
        row = {
            **{k: float(v) for k, v in bnd.items()},
            "state_norm_mean": float(X.norm(dim=-1).mean()),
            "readout_norm": float(w_r.norm()),
        }
    return row


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
    del ck["model_state"]

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

    blank = sc.observe_instances(test_ds, [0], keep_markers=False,
                                 marker_dim=marker_dim, generic_blank=generic_blank)

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

    budget_search = rsb.BudgetSearch(model, optimizer, EPS, s_cap, probe_ds,
                                     delay, marker_dim, generic_blank, fast=False)
    instrument = rom.InstrumentedStep(model, optimizer, budget_search)
    trainer = Trainer(model, env, optimizer, train_cfg)

    ep_rows: list[dict] = []
    prev_bvm = None

    def quick() -> dict:
        with torch.random.fork_rng():
            return cr.quick_probe(model, test_ds, delay, marker_dim, generic_blank)

    q0 = quick()
    p0 = online_probe(model, test_ds, delay, marker_dim, generic_blank)
    prev_bvm = p0["boundary_vs_mid"]
    ep_rows.append({"episode": 0, "frozen": False, **q0, **p0, "bvm_velocity": np.nan})
    print(f"  pre: delayed={q0['delayed_intact']:.3f} bvm={prev_bvm:+.4f}", flush=True)

    for ep in range(1, EPISODES + 1):
        s_pre = rom.probe_batch_states(model, probe_ds, delay, marker_dim, generic_blank)
        trainer.train(1)
        s_post = rom.probe_batch_states(model, probe_ds, delay, marker_dim, generic_blank)

        with torch.random.fork_rng():
            model.eval()
            w_r_probe = rg.readout_direction(model, blank, s_post.mean(0)).detach().double()
        w_rn = (w_r_probe / w_r_probe.norm() if float(w_r_probe.norm()) > 0
                else w_r_probe)
        ds_norm = float((s_post - s_pre).norm(dim=-1).mean())

        q = quick()
        prow = online_probe(model, test_ds, delay, marker_dim, generic_blank)
        row = {"episode": ep, "frozen": False, **q,
               "ds_norm": ds_norm,
               "raw_step_norm": float(budget_search.log[-1]["raw_step_norm"]),
               "alpha": float(budget_search.log[-1]["alpha"]),
               "accepted_norm": float(budget_search.log[-1]["accepted_norm"]),
               "cum_displacement": float(budget_search.log[-1]["displacement"]),
               **prow}
        if instrument.log:
            row.update(instrument.log[-1])
        bvm = prow["boundary_vs_mid"]
        row["bvm_velocity"] = (bvm - prev_bvm
                               if (bvm == bvm and prev_bvm == prev_bvm) else float("nan"))
        prev_bvm = bvm
        ep_rows.append(row)

        if ep % 10 == 0:
            pd.DataFrame(ep_rows).to_csv(run_dir / "per_episode.csv", index=False)
            print(f"  ep {ep}: delayed={q['delayed_intact']:.3f} bvm={bvm:+.4f} "
                  f"v={row['bvm_velocity']:+.4f} cum={row['cum_displacement']:.4f}",
                  flush=True)

    post_summary = cr._state_summary(
        model, test_ds, delay, marker_dim, generic_blank, seed, run_dir, "post")
    battery = cr._final_battery(model, test_ds, delay, marker_dim, generic_blank, seed)
    rubric = cr.success_rubric(pre_summary, post_summary, battery)

    pd.DataFrame(ep_rows).to_csv(run_dir / "per_episode.csv", index=False)
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

    return {
        "tag": tag, "seed": seed, "delay": delay,
        "rubric": rubric, "q_determinism_check": q_check,
        "param_deltas": cr.param_group_deltas(snap_start, cr.param_group_snapshot(model)),
        "n_episodes": EPISODES,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    BASE.mkdir(parents=True, exist_ok=True)
    (BASE / "logs").mkdir(exist_ok=True)

    if args.merge:
        rows = []
        for d in sorted(BASE.glob("*__*/per_episode.csv")):
            df = pd.read_csv(d)
            df["tag"] = d.parent.name
            rows.append(df)
        pd.concat(rows).to_csv(BASE / "cohort_per_episode.csv", index=False)
        print(f"merged {len(rows)} runs -> {BASE / 'cohort_per_episode.csv'}")
        return 0

    cohort = cohort_checkpoints()
    jobs = [(tag, p) for i, (tag, p) in enumerate(cohort)
            if i % args.num_shards == args.shard]
    print(f"shard {args.shard}/{args.num_shards}: {len(jobs)} runs: "
          f"{[int(t.split('seed')[1][:3]) for t, _ in jobs]}", flush=True)
    for tag, p in jobs:
        seed = int(tag.split("seed")[1][:3])
        run_dir = BASE / f"D800_interleaved_seed{seed:03d}__fragility"
        if (run_dir / "arm_summary.json").exists():
            print(f"skip {tag} (already done)", flush=True)
            continue
        print(f"=== {tag} ===", flush=True)
        try:
            s = run_one(tag, p, run_dir)
        except Exception:
            raise
        (run_dir / "arm_summary.json").write_text(json.dumps(s, indent=2, default=str))
        print(f"  saved {run_dir / 'arm_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
