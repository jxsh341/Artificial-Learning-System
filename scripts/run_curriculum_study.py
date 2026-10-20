#!/usr/bin/env python
"""Curriculum-scheduling study (Phase B-D): four budget-matched conditions
(A immediate-only, B delayed-only, C staged, D interleaved), multiple seeds,
intact-vs-reset probes throughout training, computed-answer retention curve,
counterfactual probe, and success-rate aggregation.

    python scripts/run_curriculum_study.py --config configs/experiments/curriculum_study.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artificial_system.experiments.runner import run_curriculum_study  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to the study YAML config")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="override config seeds")
    args = parser.parse_args()

    exp_dir = run_curriculum_study(config_path=args.config, seeds=args.seeds)
    print(f"Curriculum study complete. Results in: {exp_dir}")


if __name__ == "__main__":
    main()
