#!/usr/bin/env python
"""State-content analysis over saved checkpoints (report §7 item 3).

Runs decodability (A), answer-vs-feature (B), and causal state-substitution
(D) for every condition-seed in one or more experiment result dirs, tagging
each seed successful/failed by its recorded delayed accuracy.

  python scripts/run_state_analysis.py --results results/d800_interleaved \
      [--results results/c_frozen_gnn] [--results results/curriculum_study] \
      [--batch-size 50]

Outputs per experiment dir: state_content/state_content.csv (all rows) and
state_content/state_content_summary.json.
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
from artificial_system.evaluation import state_content as sc  # noqa: E402
from artificial_system.models.gnn_ssm import build_model  # noqa: E402
from artificial_system.utils.checkpoints import load_checkpoint  # noqa: E402


def _cond_parents(results_dir: Path) -> list[Path]:
    roots = [p for p in (results_dir, results_dir / "curriculum") if p.is_dir()]
    return sorted(
        {d for root in roots for d in root.iterdir()
         if d.is_dir() and any(d.glob("seed_*/checkpoint.pt"))}
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", action="append", required=True)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--min-delayed", type=float, default=None,
                    help="restrict to seeds with recorded delayed acc >= x (default: all)")
    args = ap.parse_args()

    total_failures = 0
    for results_dir in args.results:
        results_dir = Path(results_dir)
        out_dir = results_dir / "state_content"
        out_dir.mkdir(parents=True, exist_ok=True)
        all_rows, summaries = [], {}
        for cond_dir in _cond_parents(results_dir):
            for seed_dir in sorted(cond_dir.glob("seed_*")):
                ckpt_path = seed_dir / "checkpoint.pt"
                summary_path = seed_dir / "seed_summary.json"
                if not ckpt_path.exists() or not summary_path.exists():
                    continue
                recorded = json.loads(summary_path.read_text())
                delayed = recorded.get("delayed_test_accuracy")
                if args.min_delayed is not None and (delayed is None or delayed < args.min_delayed):
                    continue
                ckpt = load_checkpoint(ckpt_path)
                cfg = ckpt["config"]
                seed = int(ckpt["seed"])
                task_cfg, graph_cfg = cfg.get("task", {}), cfg.get("graph", {})
                model = build_model(
                    cfg, node_feature_dim=int(graph_cfg.get("node_feature_dim", 2)),
                    edge_feature_dim=0, num_actions=2, num_nodes=None,
                )
                model.load_state_dict(ckpt["model_state"])
                delay = int(task_cfg.get("delay", 3))
                marker_dim = int(task_cfg.get("marker_dim", 2))
                generic_blank = None
                if task_cfg.get("blank_mode", "no_markers") == "generic":
                    generic_blank = make_generic_graph(
                        int(task_cfg.get("num_nodes", 100)),
                        int(graph_cfg.get("node_feature_dim", 2)),
                        seed * 2000 + 2,
                    )
                # held-out instances are the test split
                gen = {
                    k: task_cfg.get(k, v)
                    for k, v in dict(
                        num_nodes=100, feature_dim=2, marker_dim=marker_dim,
                        max_path_len=3, edge_density=0.04, directed=False,
                        label_mode="adjacency", neg_min_distance=3).items()
                }
                gen["feature_dim"] = int(graph_cfg.get("node_feature_dim", gen["feature_dim"]))
                from artificial_system.data.graph_tasks import make_connectivity_dataset
                test_ds = make_connectivity_dataset(
                    int(task_cfg.get("num_test", 200)),
                    seed=int(task_cfg.get("dataset_seed", 123)) + 1, **gen,
                )
                rows, summary = sc.run_state_content(
                    model, test_ds, delay=delay, batch_size=args.batch_size,
                    marker_dim=marker_dim, generic_blank=generic_blank,
                )
                for r in rows:
                    r["condition"] = cond_dir.name
                    r["seed"] = seed
                    r["successful"] = delayed is not None and delayed >= 0.9
                all_rows.extend(rows)
                key = f"{cond_dir.name}__seed{seed:03d}"
                summaries[key] = {"recorded_delayed": delayed, **summary}
                print(f"[done] {results_dir.name}/{cond_dir.name}/seed {seed} "
                      f"(delayed={delayed if delayed is not None else float('nan'):.3f})")
        if all_rows:
            df = pd.DataFrame(all_rows)
            df.to_csv(out_dir / "state_content.csv", index=False)
            (out_dir / "state_content_summary.json").write_text(json.dumps(summaries, indent=2))

            show = df[df["analysis"] == "decode"]
            if len(show):
                print("\n--- decodability (held-out ridge) ---")
                piv = show.pivot_table(index="point", columns=["condition", "seed"], values="accuracy")
                print(piv.round(3).to_string())
            sub = df[df["analysis"] == "substitution"]
            if len(sub):
                print("\n--- state substitution (causal) ---")
                cols = [c for c in ("condition", "accuracy", "follows_substituted_state", "mean_norm_ratio") if c in sub.columns]
                print(sub[["condition", "seed"] + cols].head(len(sub)).round(3).to_string(index=False))
            print(f"\nCSV: {out_dir / 'state_content.csv'}")
        else:
            total_failures += 1
            print(f"(no checkpoint runs in {results_dir})")
    return 0 if total_failures < len(args.results) else 1


if __name__ == "__main__":
    sys.exit(main())
