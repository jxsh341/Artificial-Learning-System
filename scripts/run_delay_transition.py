#!/usr/bin/env python
"""Delay-transition analysis (report §8o, evaluation only).

§8k established the coarse picture: delay 10 retains everything (23/23),
delay 30 splits the population bimodally (information 23/23, behaviour 11/23),
delay 100 loses the information itself (0/23).  This run maps the transition
between those points with a fixed delay ladder, per checkpoint:

  [3, 10, 14, 18, 22, 26, 30, 40, 50, 65, 80, 100]

Per checkpoint x delay (no training, no parameter change):
- behavioural retention and the state-reset control;
- held-out ridge decodability of the answer at the response step, and at
  intermediate trial points (presentation, first delay step, quarter / half /
  three-quarter points) to locate where IN THE TRIAL the information dies;
- per-dimension point-biserial correlation between each state dimension and
  the label (which dimensions carry the code, and which lose it first);
- mean state norm, mean absolute margin, across-instance dispersion of the
  response-step state, and drift of the mean state from its delay-3 position
  (attractor diagnostics).

The causal substitution battery is NOT repeated here (§8k measured it at
3/10/30/100); a functional check at a newly located collapse point would be a
separate pre-registered follow-up.

  python scripts/run_delay_transition.py [--shard 0 --num-shards 6] [--merge]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artificial_system.environment.graph_environment import (  # noqa: E402
    make_generic_graph,
)
from artificial_system.evaluation import graph_classification as ge  # noqa: E402
from artificial_system.evaluation import state_content as sc  # noqa: E402
from artificial_system.experiments import coupling_rescue as cr  # noqa: E402
from artificial_system.models.gnn_ssm import build_model  # noqa: E402
from artificial_system.utils.checkpoints import load_checkpoint  # noqa: E402

DIRS = {
    "original": ROOT / "results" / "d800_interleaved" / "curriculum" / "D800_interleaved",
    "population": ROOT / "results" / "d800_population" / "curriculum" / "D800_interleaved",
}
LADDER = [3, 10, 14, 18, 22, 26, 30, 40, 50, 65, 80, 100]
INFO_THRESHOLD = 0.95
BEHAV_THRESHOLD = 0.9
CARRIER_R = 0.5
LOST_R = 0.3
# Saturated tanh dimensions have std <= ~5e-4 of floating-point noise; a
# correlation computed on that noise is meaningless (it produced spurious
# |r| up to 1.0 in the first pass).  Genuine code dimensions have std ~0.5 at
# delay 3.  The floor sits three orders of magnitude below the code and two
# above the noise.
VARIANCE_FLOOR = 0.01
ABRUPT_DECODE = 0.65


def checkpoints() -> list[tuple[str, int, Path]]:
    out = []
    for sample, base in DIRS.items():
        for seed_dir in sorted(base.glob("seed_*")):
            rec = seed_dir / "seed_summary.json"
            ck = seed_dir / "checkpoint.pt"
            if not (rec.exists() and ck.exists()):
                continue
            r = json.loads(rec.read_text())
            if r.get("delayed_test_accuracy", 0.0) < 0.9:
                continue
            out.append((sample, int(r["seed"]), ck))
    return out


def ladder_points(delay: int) -> list[str]:
    """Subsample of trial points for per-point decodability."""
    pts = ["post_present"]
    if delay >= 1:
        pts.append("delay_1")
    for frac, name in ((0.25, "delay_quarter"), (0.5, "delay_half"), (0.75, "delay_3quarter")):
        idx = max(1, int(round(delay * frac)))
        if 1 <= idx <= delay:
            pts.append(f"delay_{idx}")
    pts.append("pre_response")
    return sorted(set(pts), key=pts.index)


def pointbiserial(x: np.ndarray, y: np.ndarray) -> float:
    """|point-biserial correlation|, undefined (0.0) below the variance floor:
    a saturated dimension carries no measurable linear label information."""
    if x.std() < VARIANCE_FLOOR or y.std() < 1e-12:
        return 0.0
    return float(abs(np.corrcoef(x, y)[0, 1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/delay_transition")
    ap.add_argument("--delays", type=int, nargs="*", default=LADDER)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"delay_transition_shard{args.shard}.csv"

    if args.merge:
        parts = sorted(out_dir.glob("delay_transition_shard*.csv"))
        df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        df = df.sort_values(["sample", "seed", "delay"]).reset_index(drop=True)
        df.to_csv(out_dir / "delay_transition.csv", index=False)
        _summarise(df, out_dir)
        return 0

    ckpts = checkpoints()
    mine = [c for i, c in enumerate(ckpts) if i % args.num_shards == args.shard]
    print(f"[shard {args.shard}/{args.num_shards}] {len(mine)} of {len(ckpts)} "
          f"checkpoints x delays {args.delays}", flush=True)

    rows: list[dict] = []
    for sample, seed, ck_path in mine:
        ck = load_checkpoint(ck_path)
        cfg = ck["config"]
        task_cfg = cfg.get("task", {})
        marker_dim = int(task_cfg.get("marker_dim", 2))
        state_dim = int(cfg.get("ssm", {}).get("state_dim", 8))
        datasets = cr.build_datasets_from_cfg(cfg)
        test_ds = datasets["test"]
        model = build_model(
            cfg, node_feature_dim=int(cfg.get("graph", {}).get("node_feature_dim", 2)),
            edge_feature_dim=0, num_actions=2, num_nodes=None,
        )
        model.load_state_dict(ck["model_state"])
        model.eval()
        blank = None
        if task_cfg.get("blank_mode", "no_markers") == "generic":
            blank = make_generic_graph(
                int(task_cfg.get("num_nodes", 100)), marker_dim, seed * 2000 + 2
            )

        mean_state_ref = None  # delay-3 mean state, for the drift diagnostic
        for delay in args.delays:
            beh = ge.probe_delayed_intact_vs_reset(
                model, test_ds, delay=delay, batch_size=32,
                marker_dim=marker_dim, generic_blank=blank,
            )
            states = sc.capture_trial_states(
                model, test_ds, delay=delay, batch_size=args.batch_size,
                marker_dim=marker_dim, generic_blank=blank,
            )
            model.eval()
            X = states["pre_response"].numpy()
            labels = states["_label"].numpy().astype(int)
            dec = sc.ridge_decode_cv(states["pre_response"].numpy(), labels, seed=seed)
            pts = ladder_points(delay)
            point_dec = sc.decode_at_points(states, pts, seed=seed)

            mean_state = X.mean(0)
            drift = float("nan")
            if mean_state_ref is not None:
                drift = float(np.linalg.norm(mean_state - mean_state_ref))
            if delay == args.delays[0]:
                mean_state_ref = mean_state

            row = {
                "sample": sample, "seed": seed, "delay": int(delay),
                "delayed_intact": float(beh["delayed_intact"]),
                "delayed_reset": float(beh["delayed_state_reset"]),
                "decode_pre_response": float(dec["accuracy"]),
                "decode_baseline": float(dec["majority_baseline"]),
                "mean_state_norm": float(np.linalg.norm(X, axis=1).mean()),
                "mean_abs_margin": float(np.abs(states["_signed_margin"].numpy()).mean()),
                "across_instance_std": float(X.std(axis=0).mean()),
                "drift_from_first_delay": drift,
            }
            for p in pts:
                row[f"decode_{p}"] = float(point_dec[p]["accuracy"])
            for j in range(state_dim):
                row[f"r_dim{j}"] = pointbiserial(X[:, j], labels.astype(float))
                row[f"std_dim{j}"] = float(X[:, j].std())
            rows.append(row)
            print(f"  {sample:>10} seed {seed:>2} delay {delay:>3}: "
                  f"behav={row['delayed_intact']:.3f} decode={row['decode_pre_response']:.3f} "
                  f"reset={row['delayed_reset']:.3f} drift={row['drift_from_first_delay']:.3f} "
                  f"disp={row['across_instance_std']:.3f}", flush=True)
            pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"wrote {csv_path}", flush=True)
    return 0


def _summarise(df: pd.DataFrame, out_dir: Path) -> None:
    state_cols = [c for c in df.columns if c.startswith("r_dim")]
    delays = sorted(df.delay.unique())

    per_ckpt: dict = {}
    for (sample, seed), g in df.groupby(["sample", "seed"]):
        g = g.sort_values("delay")
        info_below = g[g.decode_pre_response < INFO_THRESHOLD]
        beh_below = g[g.delayed_intact < BEHAV_THRESHOLD]
        d_info = int(info_below.delay.iloc[0]) if len(info_below) else None
        d_behav = int(beh_below.delay.iloc[0]) if len(beh_below) else None
        dec_at_info = (
            float(g.loc[g.delay == d_info, "decode_pre_response"].iloc[0])
            if d_info is not None else None
        )
        abrupt = bool(dec_at_info is not None and dec_at_info <= ABRUPT_DECODE)
        g3 = g[g.delay == args_delays0(df)]
        carriers = [int(c[-1]) for c in state_cols
                    if len(g3) and float(g3.iloc[0][c]) >= CARRIER_R]
        # carrier loss: only dims that were carriers at the reference delay;
        # lost = first delay with |r| < LOST_R or variance below the floor
        carrier_loss = {}
        for dim in carriers:
            rcol, scol = f"r_dim{dim}", f"std_dim{dim}"
            lost = g[(g[rcol] < LOST_R) | (g[scol] < VARIANCE_FLOOR)]
            if len(lost):
                carrier_loss[f"dim{dim}"] = int(lost.delay.iloc[0])
        per_ckpt[f"{sample}_seed{seed:03d}"] = {
            "D_info": d_info,
            "D_behavioural": d_behav,
            "decode_at_D_info": dec_at_info,
            "abrupt": abrupt,
            "carrier_dims_at_delay3": carriers,
            "carrier_loss_delay": carrier_loss,
            "decode_trajectory": {int(r.delay): float(r.decode_pre_response) for r in g.itertuples()},
            "behav_trajectory": {int(r.delay): float(r.delayed_intact) for r in g.itertuples()},
            "drift_trajectory": {
                int(r.delay): (None if r.drift_from_first_delay != r.drift_from_first_delay
                               else float(r.drift_from_first_delay)) for r in g.itertuples()
            },
            "dispersion_trajectory": {
                int(r.delay): float(r.across_instance_std) for r in g.itertuples()
            },
        }

    d_infos = [v["D_info"] for v in per_ckpt.values() if v["D_info"] is not None]
    d_behavs = [v["D_behavioural"] for v in per_ckpt.values() if v["D_behavioural"] is not None]
    summary = {
        "ladder": [int(d) for d in delays],
        "n_checkpoints": int(df.groupby(["sample", "seed"]).ngroups),
        "thresholds": {
            "information": INFO_THRESHOLD, "behavioural": BEHAV_THRESHOLD,
            "abrupt_decode": ABRUPT_DECODE, "carrier_r": CARRIER_R, "lost_r": LOST_R,
            "variance_floor": VARIANCE_FLOOR,
        },
        "D_info_distribution": {str(d): d_infos.count(d) for d in sorted(set(d_infos))},
        "D_behavioural_distribution": {str(d): d_behavs.count(d) for d in sorted(set(d_behavs))},
        "n_abrupt": int(sum(v["abrupt"] for v in per_ckpt.values())),
        "n_gradual": int(sum(not v["abrupt"] and v["D_info"] is not None
                             for v in per_ckpt.values())),
        "behaviour_before_information": int(
            sum(1 for v in per_ckpt.values()
                if v["D_behavioural"] is not None and v["D_info"] is not None
                and v["D_behavioural"] < v["D_info"])
        ),
        "per_checkpoint": per_ckpt,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    print(f"\nD_info distribution: {summary['D_info_distribution']}")
    print(f"D_behavioural distribution: {summary['D_behavioural_distribution']}")
    print(f"abrupt: {summary['n_abrupt']}, gradual: {summary['n_gradual']}, "
          f"behaviour fails before information: {summary['behaviour_before_information']}")
    print(f"wrote {out_dir / 'delay_transition.csv'} and {out_dir / 'summary.json'}")


def args_delays0(df: pd.DataFrame) -> int:
    """The first (reference) delay of the ladder, as stored in the CSV."""
    return int(df.delay.min())


if __name__ == "__main__":
    sys.exit(main())
