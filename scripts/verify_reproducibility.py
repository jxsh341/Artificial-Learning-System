#!/usr/bin/env python
"""Reproducibility verification (pre-launch milestone).

For every condition-seed of the curriculum study:
  1. load the saved checkpoint + the config stored inside it;
  2. rebuild the test dataset and model from that config alone;
  3. re-run the delayed and immediate probes deterministically;
  4. compare against the recorded seed_summary metrics;
  5. re-aggregate all per-seed summaries and diff against summary.json.

Exit code 1 if any check fails.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import yaml  # noqa: E402

from artificial_system.data.graph_tasks import make_connectivity_dataset  # noqa: E402
from artificial_system.environment.graph_environment import GraphClassificationEnvironment  # noqa: E402
from artificial_system.evaluation import graph_classification as graph_eval  # noqa: E402
from artificial_system.experiments.runner import load_config  # noqa: E402
from artificial_system.models.gnn_ssm import build_model  # noqa: E402
from artificial_system.utils.checkpoints import load_checkpoint  # noqa: E402

RESULTS = ROOT / "results" / "curriculum_study"
TOL = 1e-6


def rebuild_envs(cfg: dict, seed: int):
    """Rebuild train-env (for the generic blank graph) and datasets from config."""
    task_cfg = cfg.get("task", {})
    gen_cfg = dict(
        num_nodes=int(task_cfg.get("num_nodes", 100)),
        feature_dim=int(cfg.get("graph", {}).get("node_feature_dim", 16)),
        marker_dim=int(task_cfg.get("marker_dim", 2)),
        max_path_len=int(task_cfg.get("max_path_len", 3)),
        edge_density=float(task_cfg.get("edge_density", 0.05)),
        directed=bool(task_cfg.get("directed", False)),
        label_mode=task_cfg.get("label_mode", "adjacency"),
        neg_min_distance=int(task_cfg.get("neg_min_distance", 3)),
    )
    ds_seed = int(task_cfg.get("dataset_seed", 123))
    train_ds = make_connectivity_dataset(int(task_cfg.get("num_train", 2000)), seed=ds_seed, **gen_cfg)
    test_ds = make_connectivity_dataset(int(task_cfg.get("num_test", 200)), seed=ds_seed + 1, **gen_cfg)
    env_del = GraphClassificationEnvironment(train_ds, task_cfg, seed=seed * 2000 + 2)
    return test_ds, env_del


def main() -> int:
    failures = 0
    checked = 0
    for cond_dir in sorted((RESULTS / "curriculum").iterdir()):
        if not cond_dir.is_dir():
            continue
        for seed_dir in sorted(cond_dir.glob("seed_*")):
            ckpt_path = seed_dir / "checkpoint.pt"
            summary_path = seed_dir / "seed_summary.json"
            if not ckpt_path.exists() or not summary_path.exists():
                print(f"FAIL {cond_dir.name}/{seed_dir.name}: missing checkpoint or summary")
                failures += 1
                continue
            recorded = json.loads(summary_path.read_text())
            ckpt = load_checkpoint(ckpt_path)
            cfg = ckpt["config"]
            seed = int(ckpt["seed"])

            model = build_model(
                cfg,
                node_feature_dim=int(cfg.get("graph", {}).get("node_feature_dim", 2)),
                edge_feature_dim=0,
                num_actions=2,
                num_nodes=None,
            )
            model.load_state_dict(ckpt["model_state"])
            model.eval()

            test_ds, env_del = rebuild_envs(cfg, seed)
            delay = int(cfg.get("task", {}).get("delay", 3))
            marker_dim = int(cfg.get("task", {}).get("marker_dim", 2))
            eval_batch = int(cfg.get("evaluation", {}).get("eval_batch_size", 16))
            generic_blank = (
                env_del._generic_graph
                if getattr(env_del, "blank_mode", "no_markers") == "generic"
                else None
            )

            probe_del = graph_eval.probe_dataset(
                model, test_ds, delay=delay, batch_size=eval_batch,
                marker_dim=marker_dim, generic_blank=generic_blank,
            )
            probe_imm = graph_eval.probe_dataset(
                model, test_ds, delay=0, batch_size=eval_batch, marker_dim=marker_dim,
            )

            ok_del = abs(probe_del["accuracy"] - recorded["delayed_test_accuracy"]) < TOL
            ok_imm = abs(probe_imm["accuracy"] - recorded["immediate_test_accuracy"]) < TOL
            checked += 1
            status = "OK  " if (ok_del and ok_imm) else "FAIL"
            if not (ok_del and ok_imm):
                failures += 1
            print(f"{status} {cond_dir.name}/{seed_dir.name}: "
                  f"delayed {probe_del['accuracy']:.4f} vs {recorded['delayed_test_accuracy']:.4f} | "
                  f"immediate {probe_imm['accuracy']:.4f} vs {recorded['immediate_test_accuracy']:.4f}")

    # re-aggregate per-seed summaries and diff against the committed aggregate
    agg_path = RESULTS / "summary.json"
    if agg_path.exists():
        committed = json.loads(agg_path.read_text())
        for cond, d in committed.items():
            per_seed = sorted(
                (RESULTS / "curriculum" / cond).glob("seed_*/seed_summary.json"),
                key=lambda p: int(p.parent.name.split("_")[1]),
            )
            recs = [json.loads(p.read_text()) for p in per_seed]
            accs = [round(r["delayed_test_accuracy"], 6) for r in recs]
            committed_accs = [round(x["delayed_test_accuracy"], 6) for x in d["per_seed"]]
            if accs != committed_accs:
                print(f"FAIL aggregate mismatch for {cond}: {accs} vs {committed_accs}")
                failures += 1
            else:
                print(f"OK   aggregate {cond}: per-seed accs match ({accs})")
    else:
        print("FAIL summary.json missing")
        failures += 1

    print(f"\n{checked} condition-seeds re-evaluated; {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
