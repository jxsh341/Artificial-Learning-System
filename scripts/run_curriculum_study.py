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
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artificial_system.experiments.runner import run_curriculum_study  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to the study YAML config")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="override config seeds")
    args = parser.parse_args()

    # Always announce the run up front and flush: when stdout is piped or
    # redirected (e.g. `| Select-Object -Last 3`, `> run.log`) the default
    # block buffering makes the process look silent for the whole run.
    t0 = time.time()
    print(f"[run_curriculum_study] config={args.config} seeds={args.seeds or 'config'}", flush=True)
    print(f"[run_curriculum_study] started {time.strftime('%H:%M:%S')}", flush=True)

    exp_dir = run_curriculum_study(config_path=args.config, seeds=args.seeds)

    mins = (time.time() - t0) / 60.0
    print(f"[run_curriculum_study] finished in {mins:.1f} min", flush=True)
    print(f"Curriculum study complete. Results in: {exp_dir}", flush=True)


if __name__ == "__main__":
    main()
