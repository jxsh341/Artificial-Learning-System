#!/usr/bin/env python
"""Evaluate a saved checkpoint (retention, generalization) without retraining.

    python scripts/evaluate.py --config configs/base.yaml --graph small_world \
        --checkpoint results/base/seed_000/checkpoint.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artificial_system.experiments.runner import run_evaluation_only  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to the experiment YAML config")
    parser.add_argument("--checkpoint", required=True, help="path to a checkpoint .pt file")
    parser.add_argument("--graph", default=None, help="graph source (overrides config)")
    args = parser.parse_args()

    exp_dir = run_evaluation_only(
        args.config, checkpoint_path=args.checkpoint, graph_source=args.graph
    )
    print(f"Evaluation complete. Results in: {exp_dir}")


if __name__ == "__main__":
    main()
