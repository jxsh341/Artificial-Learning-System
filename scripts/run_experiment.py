#!/usr/bin/env python
"""Run a full experiment (Definition of Done, spec §30):

    python scripts/run_experiment.py --config configs/base.yaml --graph <graph_source>

Loads + validates the graph, initializes GNN + SSM, trains, checkpoints,
runs ARRG evaluation and configured ablations, and exports metrics.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artificial_system.experiments.runner import run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to the experiment YAML config")
    parser.add_argument(
        "--graph",
        default=None,
        help="graph source: synthetic family (random|regular|small_world|"
        "scale_free|modular) or path to an .npz export (overrides config)",
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="override config seeds")
    args = parser.parse_args()

    exp_dir = run_experiment(
        config_path=args.config, graph_source=args.graph, seeds=args.seeds
    )
    print(f"Experiment complete. Results in: {exp_dir}")


if __name__ == "__main__":
    main()
