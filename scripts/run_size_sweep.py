#!/usr/bin/env python
"""Controlled size-generalization sweep (TECHNICAL REPORT §7 item 2).

Evaluates *saved checkpoints* (no retraining) across a ladder of graph sizes
with node count isolated from mean degree:

  python scripts/run_size_sweep.py --results results/d800_interleaved \
      [--sizes 100,130,160,200] [--sets 2] [--instances 200] \
      [--min-delayed 0.9]

For every condition-seed whose recorded delayed accuracy meets
``--min-delayed`` (i.e. only *successful* runs), per-size rows are written to
``<results>/size_sweep/size_sweep.csv`` plus a per-seed aggregate JSON summary.
Node-count dependence is measured under two density modes per size:
``fixed`` (training density; size and degree confounded) and
``degree_matched`` (density scaled proportional to base_n/N).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from artificial_system.environment.graph_environment import (  # noqa: E402
    make_generic_graph,
)
from artificial_system.evaluation import size_sweep as sw  # noqa: E402
from artificial_system.models.gnn_ssm import build_model  # noqa: E402
from artificial_system.utils.checkpoints import load_checkpoint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="experiment results dir (contains condition/seed dirs)")
    ap.add_argument("--out", default=None, help="output dir (default <results>/size_sweep)")
    ap.add_argument("--sizes", default="100,130,160,200")
    ap.add_argument("--sets", type=int, default=2)
    ap.add_argument("--instances", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--min-delayed", type=float, default=0.9)
    args = ap.parse_args()

    results_dir = Path(args.results)
    out_dir = Path(args.out) if args.out else results_dir / "size_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = [int(s) for s in args.sizes.split(",")]

    roots = [p for p in (results_dir, results_dir / "curriculum") if p.is_dir()]
    parents = sorted(
        {d for root in roots for d in root.iterdir()
         if d.is_dir() and any(d.glob("seed_*/checkpoint.pt"))}
    )

    all_rows: list[dict] = []
    seed_summaries: dict[str, dict] = {}

    for cond_dir in parents:
        for seed_dir in sorted(cond_dir.glob("seed_*")):
            ckpt_path = seed_dir / "checkpoint.pt"
            summary_path = seed_dir / "seed_summary.json"
            if not ckpt_path.exists() or not summary_path.exists():
                continue
            recorded = json.loads(summary_path.read_text())
            delayed = recorded.get("delayed_test_accuracy")
            if delayed is None or delayed < args.min_delayed:
                continue

            ckpt = load_checkpoint(ckpt_path)
            cfg = ckpt["config"]
            seed = int(ckpt["seed"])
            task_cfg = cfg.get("task", {})
            graph_cfg = cfg.get("graph", {})

            model = build_model(
                cfg,
                node_feature_dim=int(graph_cfg.get("node_feature_dim", 2)),
                edge_feature_dim=0,
                num_actions=2,
                num_nodes=None,
            )
            model.load_state_dict(ckpt["model_state"])

            # instance-independent blank graph (matches blank_mode: generic
            # training; same construction the env uses, applied directly so
            # no dataset is required)
            generic_blank = None
            if task_cfg.get("blank_mode", "no_markers") == "generic":
                # same base seed the trainer used for the env (see
                # run_curriculum_condition): seed * 2000 + 2
                generic_blank = make_generic_graph(
                    int(task_cfg.get("num_nodes", 100)),
                    int(graph_cfg.get("node_feature_dim", 2)),
                    seed * 2000 + 2,
                )

            base_gen = sw._gen_cfg(task_cfg, graph_cfg)
            base_gen["base_nodes"] = int(task_cfg.get("num_nodes", sizes[0]))
            delay = int(task_cfg.get("delay", 3))

            rows = sw.sweep_size(
                model,
                task_cfg=None,
                graph_cfg=None,
                sizes=sizes,
                sets_per_size=args.sets,
                instances_per_set=args.instances,
                delay=delay,
                batch_size=args.batch_size,
                generic_blank_graph=generic_blank,
                base_gen=base_gen,
            )
            for r in rows:
                r["condition"] = cond_dir.name
                r["seed"] = seed
                r["trained_delay"] = delay
            all_rows.extend(rows)

            agg = (
                pd.DataFrame(rows)
                .groupby(["num_nodes", "density_mode"])
                .agg(
                    immediate=("immediate_accuracy", "mean"),
                    delayed_intact=("delayed_intact", "mean"),
                    delayed_reset=("delayed_reset", "mean"),
                    z_norm=("mean_z_norm", "mean"),
                    h_norm=("mean_h_norm", "mean"),
                    marker_share=("mean_marker_attention_share", "mean"),
                )
                .reset_index()
            )
            seed_summaries[f"{cond_dir.name}_seed{seed:03d}"] = {
                "delayed_test_accuracy": delayed,
                "per_size": agg.to_dict(orient="records"),
            }
            print(f"[done] {cond_dir.name}/seed {seed} (delayed={delayed:.3f}) {len(rows)} rows")

    if not all_rows:
        print("No qualifying successful seeds found.")
        return 1
    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "size_sweep.csv"
    df.to_csv(csv_path, index=False)
    (out_dir / "size_sweep_summary.json").write_text(json.dumps(seed_summaries, indent=2))

    mega = (
        df.groupby(["num_nodes", "density_mode"])
        .agg(
            immediate_mean=("immediate_accuracy", "mean"),
            immediate_min=("immediate_accuracy", "min"),
            intact_mean=("delayed_intact", "mean"),
            intact_min=("delayed_intact", "min"),
            reset_mean=("delayed_reset", "mean"),
            z_norm=("mean_z_norm", "mean"),
            h_norm=("mean_h_norm", "mean"),
            marker_share=("mean_marker_attention_share", "mean"),
            margin=("mean_logit_margin", "mean"),
            n_rows=("immediate_accuracy", "count"),
        )
        .reset_index()
    )
    mega.to_csv(out_dir / "overall_by_size.csv", index=False)
    print("\n=== overall (mean over all qualifying seeds) ===")
    print(mega.to_string(index=False))
    print(f"\nCSV: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
