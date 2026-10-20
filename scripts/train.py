#!/usr/bin/env python
"""Train the combined model on the cue task (no ARRG evaluation).

    python scripts/train.py --config configs/base.yaml --graph small_world --seeds 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artificial_system.experiments.runner import run_training_only  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to the experiment YAML config")
    parser.add_argument("--graph", default=None, help="graph source (overrides config)")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="override config seeds")
    args = parser.parse_args()

    exp_dir = run_training_only(args.config, graph_source=args.graph, seeds=args.seeds)
    print(f"Training complete. Results in: {exp_dir}")


if __name__ == "__main__":
    main()
