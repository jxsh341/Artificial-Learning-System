#!/usr/bin/env python
"""§8f: per-group encoder gradient conflict on the D-int s0 captured state.

  python scripts/run_encoder_gradient_diagnostics.py \
      --captured results/coupling_rescue_h7/D_interleaved_seed000__h7_capture_hold/captured_checkpoint.pt \
      --episodes 200

Runs the H7 stage-4 regime (frozen readout, joint episodes) from the captured
checkpoint, logging per-group (gnn / pooling / ssm_gen) delayed-vs-immediate
gradient norms, cosines, combined norms, margins, and behavioural probes every
5 episodes. The source checkpoint is only read.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artificial_system.experiments import coupling_rescue as cr  # noqa: E402
from artificial_system.experiments import readout_diagnostics as rd  # noqa: E402
from artificial_system.utils.checkpoints import load_checkpoint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captured", required=True)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--results-dir", default="results/encoder_gradient_diagnostics")
    args = ap.parse_args()

    cap = torch.load(ROOT / args.captured, weights_only=False)
    cfg = cap["config"]
    seed = int(cap["seed"])
    datasets = cr.build_datasets_from_cfg(cfg)

    out_dir = Path(args.results_dir) / f"D_interleaved_seed{seed:03d}"
    summary = rd.run_encoder_gradient_diagnostics(
        cfg, datasets, cap, out_dir, seed=seed,
        episodes=args.episodes,
    )
    print(json.dumps(summary, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
