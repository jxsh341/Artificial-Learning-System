#!/usr/bin/env python
"""Track-2 diagnostic: readout-path convergence on D_interleaved seed 0 (§8d).

  python scripts/run_readout_diagnostics.py --source D_interleaved:0 \
      --episodes 100 --lrs 0.0003,0.001,0.003 --restarts 4

For each (LR, restart): readout-only delayed training from an untouched copy
of the failing checkpoint, with per-episode gradient-conflict logging.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artificial_system.experiments import coupling_rescue as cr
from artificial_system.experiments import readout_diagnostics as rd
from artificial_system.experiments.runner import load_config
from artificial_system.utils.checkpoints import load_checkpoint

STUDY_RESULTS = ROOT / "results" / "curriculum_study"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/curriculum_study.yaml")
    ap.add_argument("--source", required=True, help="condition:seed, e.g. D_interleaved:0")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--lrs", default="0.0003,0.001,0.003")
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--results-dir", default="results/readout_diagnostics")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    datasets = cr.build_datasets_from_cfg(cfg)

    cond, seed_s = args.source.split(":")
    seed = int(seed_s)
    ck_path = STUDY_RESULTS / "curriculum" / cond / f"seed_{seed:03d}" / "checkpoint.pt"
    recorded = json.loads((ck_path.parent / "seed_summary.json").read_text())
    out_dir = Path(args.results_dir) / f"{cond}_seed{seed:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for lr in [float(x) for x in args.lrs.split(",")]:
        for restart in range(args.restarts):
            ck = load_checkpoint(ck_path)  # fresh copy every time
            ck["recorded_delayed"] = recorded.get("delayed_test_accuracy")
            s = rd.run_readout_diagnostics(
                cfg, datasets, ck, out_dir, seed=seed,
                episodes=args.episodes, learning_rate=lr,
                restart_offset=restart * 7919,
            )
            s["restart"] = restart
            summaries[f"lr{lr:g}_r{restart}"] = s
            print(f"lr={lr:g} r{restart}: delayed {s['delayed_final']:.3f} "
                  f"cos(tail) {s['cos_imm_delayed_mean_tail']:+.3f} "
                  f"loss {s['loss_delayed_initial']:.3f}->{s['loss_delayed_final']:.3f}")
    (out_dir / "diag_summary.json").write_text(json.dumps(summaries, indent=2, default=float))
    print(f"\nresults: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
