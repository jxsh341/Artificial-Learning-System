#!/usr/bin/env python
"""Evaluation-only delay scaling (report §8k).

For every successful D800 checkpoint (original 3 + population 20) evaluate the
*trained* model at delays 3, 10, 30 and 100 with **no training and no parameter
change**, and report the three retention outcomes separately:

- behavioural : ``delayed_intact`` accuracy (and the state-reset control)
- information : held-out linear-probe decodability of the answer at the response
                step, plus mean state norm and mean response logit margin
- functional  : causal state substitution (same-answer, different-answer
                follows, zero and shuffled controls)

  python scripts/run_delay_scaling.py [--delays 3 10 30 100] \
      [--shard 0 --num-shards 4]

Outputs results/delay_scaling/delay_scaling.csv (one row per checkpoint x delay)
and summary.json (per-seed trajectories; no pooling in the headline).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


def _checkpoints() -> list[tuple[str, int, Path, dict]]:
    out = []
    for sample, base in DIRS.items():
        for seed_dir in sorted(base.glob("seed_*")):
            rec_path = seed_dir / "seed_summary.json"
            ck_path = seed_dir / "checkpoint.pt"
            if not (rec_path.exists() and ck_path.exists()):
                continue
            rec = json.loads(rec_path.read_text())
            if rec.get("delayed_test_accuracy", 0.0) < 0.9:
                continue
            out.append((sample, int(rec["seed"]), ck_path, rec))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delays", type=int, nargs="*", default=[3, 10, 30, 100])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--results-dir", default="results/delay_scaling")
    args = ap.parse_args()

    out_dir = ROOT / args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"delay_scaling_shard{args.shard}.csv"

    ckpts = _checkpoints()
    mine = [c for i, c in enumerate(ckpts) if i % args.num_shards == args.shard]
    print(f"[shard {args.shard}/{args.num_shards}] {len(mine)} of {len(ckpts)} checkpoints",
          flush=True)

    rows: list[dict] = []
    for sample, seed, ck_path, rec in mine:
        ck = load_checkpoint(ck_path)
        cfg = ck["config"]
        task_cfg = cfg.get("task", {})
        marker_dim = int(task_cfg.get("marker_dim", 2))
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
            blank = make_generic_graph(int(task_cfg.get("num_nodes", 100)), marker_dim,
                                       seed * 2000 + 2)

        for delay in args.delays:
            probe = ge.probe_delayed_intact_vs_reset(
                model, test_ds, delay=delay, batch_size=32,
                marker_dim=marker_dim, generic_blank=blank,
            )
            states = sc.capture_trial_states(
                model, test_ds, delay=delay, batch_size=args.batch_size,
                marker_dim=marker_dim, generic_blank=blank,
            )
            dec = sc.decode_at_points(states, ["pre_response"], seed=seed)["pre_response"]
            _, subst = sc.state_substitution(
                model, test_ds, delay=delay, batch_size=args.batch_size,
                marker_dim=marker_dim, generic_blank=blank, seed=seed,
            )
            row = {
                "sample": sample,
                "seed": seed,
                "delay": int(delay),
                "delayed_intact": float(probe["delayed_intact"]),
                "delayed_reset": float(probe["delayed_state_reset"]),
                "decode_pre_response": float(dec["accuracy"]),
                "decode_baseline": float(dec["majority_baseline"]),
                "mean_state_norm": float(states["pre_response"].norm(dim=-1).mean()),
                "mean_logit_margin": float(states["_margin"].mean()),
                "subst_same_follows": float(subst["subst_same_follows"]),
                "subst_diff_follows": float(subst["subst_diff_follows"]),
                "subst_zero_accuracy": float(subst["subst_zero_accuracy"]),
                "subst_shuffled_accuracy": float(subst["subst_shuffled_accuracy"]),
            }
            rows.append(row)
            print(f"  {sample:>10} seed {seed:>2} delay {delay:>3}: "
                  f"behav={row['delayed_intact']:.3f} decode={row['decode_pre_response']:.3f} "
                  f"follows={row['subst_diff_follows']:.3f} "
                  f"zero={row['subst_zero_accuracy']:.3f}", flush=True)
            pd.DataFrame(rows).to_csv(csv_path, index=False)

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"wrote {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
