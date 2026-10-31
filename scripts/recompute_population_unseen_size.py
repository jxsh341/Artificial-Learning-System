#!/usr/bin/env python
"""Recompute corrected unseen-size accuracy for every D800 population seed.

The original 3-seed run predates the probe-protocol fix (generic blank during
delay), so its recorded ``graph_generalization_unseen_size`` values are legacy
artifacts (0.505). The 20 population seeds ran post-fix, but this script
recomputes ALL 23 with one identical protocol so the reported table is
internally consistent and no stale value is carried forward.

Protocol: n=200 test graphs, density 0.04 (matching the frozen task config),
dataset seed 125 (dataset_seed+2), delay 3, generic blank constructed with the
checkpoint's own num_nodes and seed*2000+2.

Writes results/d800_population/unseen_size_corrected.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artificial_system.data.graph_tasks import make_connectivity_dataset  # noqa: E402
from artificial_system.environment.graph_environment import make_generic_graph  # noqa: E402
from artificial_system.evaluation import graph_classification as ge  # noqa: E402
from artificial_system.models.gnn_ssm import build_model  # noqa: E402
from artificial_system.utils.checkpoints import load_checkpoint  # noqa: E402

DIRS = {
    "original": ROOT / "results" / "d800_interleaved" / "curriculum" / "D800_interleaved",
    "population": ROOT / "results" / "d800_population" / "curriculum" / "D800_interleaved",
}

ds_gen = dict(num_nodes=200, feature_dim=2, marker_dim=2, max_path_len=3,
              edge_density=0.04, directed=False, label_mode="adjacency",
              neg_min_distance=3)
unseen = make_connectivity_dataset(200, seed=125, **ds_gen)


def main() -> int:
    rows = []
    for sample, base in DIRS.items():
        for seed_dir in sorted(base.glob("seed_*")):
            rec = json.loads((seed_dir / "seed_summary.json").read_text())
            ckpt = load_checkpoint(seed_dir / "checkpoint.pt")
            cfg = ckpt["config"]
            seed = int(ckpt["seed"])
            model = build_model(cfg, node_feature_dim=2, edge_feature_dim=0,
                                num_actions=2, num_nodes=None)
            model.load_state_dict(ckpt["model_state"])
            blank = make_generic_graph(int(cfg["task"]["num_nodes"]), 2, seed * 2000 + 2)
            acc = ge.probe_dataset(model, unseen, delay=3, batch_size=16,
                                   marker_dim=2, generic_blank=blank)["accuracy"]
            rows.append({
                "sample": sample, "seed": seed,
                "recorded": rec.get("graph_generalization_unseen_size"),
                "corrected": float(acc),
                "delayed": rec.get("delayed_test_accuracy"),
                "immediate": rec.get("immediate_test_accuracy"),
            })
            print(f"{sample:>10} seed {seed:>2}: recorded="
                  f"{rec.get('graph_generalization_unseen_size', float('nan')):.3f} "
                  f"-> corrected={acc:.3f}")

    out = ROOT / "results" / "d800_population" / "unseen_size_corrected.json"
    out.write_text(json.dumps({"description": __doc__, "rows": rows}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
