#!/usr/bin/env python
"""Recompute corrected delayed unseen-size metrics for the curriculum-study
successful seeds (probe-protocol fix: generic blank during delay).

Writes results/curriculum_study/unseen_size_corrected.json and prints a table.
Successful seeds = recorded delayed_test_accuracy >= 0.9.
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

RESULTS = ROOT / "results" / "curriculum_study"
ds_gen = dict(num_nodes=200, feature_dim=2, marker_dim=2, max_path_len=3,
              edge_density=0.04, directed=False, label_mode="adjacency",
              neg_min_distance=3)
unseen = make_connectivity_dataset(200, seed=125, **ds_gen)

rows = []
for cond_dir in sorted((RESULTS / "curriculum").iterdir()):
    if not cond_dir.is_dir():
        continue
    for seed_dir in sorted(cond_dir.glob("seed_*")):
        rec = json.loads((seed_dir / "seed_summary.json").read_text())
        if rec.get("delayed_test_accuracy", 0.0) < 0.9:
            continue
        ckpt = load_checkpoint(seed_dir / "checkpoint.pt")
        cfg = ckpt["config"]
        seed = int(ckpt["seed"])
        model = build_model(cfg, node_feature_dim=2, edge_feature_dim=0,
                            num_actions=2, num_nodes=None)
        model.load_state_dict(ckpt["model_state"])
        old = rec["graph_generalization_unseen_size"]
        blank = make_generic_graph(int(cfg["task"]["num_nodes"]), 2, seed * 2000 + 2)
        new = ge.probe_dataset(model, unseen, delay=3, batch_size=16,
                               marker_dim=2, generic_blank=blank)["accuracy"]
        rows.append({"condition": cond_dir.name, "seed": seed,
                     "recorded_legacy": old, "corrected_generic_blank": new})
        print(f"{cond_dir.name} seed {seed}: legacy={old:.3f} -> corrected={new:.3f}")

out = RESULTS / "unseen_size_corrected.json"
out.write_text(json.dumps({"description": __doc__, "rows": rows}, indent=2))
print(f"\nwrote {out}")
