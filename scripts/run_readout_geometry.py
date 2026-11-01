#!/usr/bin/env python
"""Stage A of §8l: readout–probe geometry, measured BEFORE any intervention.

For every successful D800 checkpoint, at delays 3, 10 and 30, in the same raw
state coordinate system:

- held-out probe decodability at the response step (information present?);
- cosine between the probe direction and the effective readout direction
  ``d(logit1 - logit0)/ds`` (rotation?);
- per-class readout accuracy and the normalised boundary position
  (``boundary_vs_mid``; outside +/-0.5 means one class is entirely on the wrong
  side of the fixed boundary = translation, not rotation);
- projections onto each direction, signed/absolute margin, state norm,
  probe–readout prediction agreement.

No parameters are changed.  The output is the stage-A prediction that stage B
(H7b at delay 30) will test.

  python scripts/run_readout_geometry.py [--delays 3 10 30]

Outputs results/readout_geometry/geometry.csv and summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artificial_system.environment.graph_environment import (  # noqa: E402
    make_generic_graph,
)
from artificial_system.evaluation import graph_classification as ge  # noqa: E402
from artificial_system.evaluation import readout_geometry as rg  # noqa: E402
from artificial_system.experiments import coupling_rescue as cr  # noqa: E402
from artificial_system.models.gnn_ssm import build_model  # noqa: E402
from artificial_system.utils.checkpoints import load_checkpoint  # noqa: E402

DIRS = {
    "original": ROOT / "results" / "d800_interleaved" / "curriculum" / "D800_interleaved",
    "population": ROOT / "results" / "d800_population" / "curriculum" / "D800_interleaved",
}


def checkpoints() -> list[tuple[str, int, Path]]:
    out = []
    for sample, base in DIRS.items():
        for seed_dir in sorted(base.glob("seed_*")):
            rec_path = seed_dir / "seed_summary.json"
            ck_path = seed_dir / "checkpoint.pt"
            if not (rec_path.exists() and ck_path.exists()):
                continue
            rec = json.loads(rec_path.read_text())
            if rec.get("delayed_test_accuracy", 0.0) < 0.9:
                continue
            out.append((sample, int(rec["seed"]), ck_path))
    return out


def geometry_class(row: dict) -> str:
    """Descriptive label for one checkpoint x delay."""
    if row["decode_accuracy"] < 0.95:
        return "information_absent"
    if row["cos_probe_readout"] < 0.90:
        return "rotated"
    if abs(row["boundary_vs_mid"]) > 0.5:
        return "translated"
    return "aligned"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delays", type=int, nargs="*", default=[3, 10, 30])
    ap.add_argument("--results-dir", default="results/readout_geometry")
    args = ap.parse_args()

    out_dir = ROOT / args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpts = checkpoints()
    print(f"[geometry] {len(ckpts)} successful checkpoints x delays {args.delays}", flush=True)

    rows: list[dict] = []
    for sample, seed, ck_path in ckpts:
        ck = load_checkpoint(ck_path)
        cfg = ck["config"]
        task_cfg = cfg.get("task", {})
        marker_dim = int(task_cfg.get("marker_dim", 2))
        datasets = cr.build_datasets_from_cfg(cfg)
        model = build_model(
            cfg, node_feature_dim=int(cfg.get("graph", {}).get("node_feature_dim", 2)),
            edge_feature_dim=0, num_actions=2, num_nodes=None,
        )
        model.load_state_dict(ck["model_state"])
        model.eval()
        blank = None
        if task_cfg.get("blank_mode", "no_markers") == "generic":
            blank = make_generic_graph(
                int(task_cfg.get("num_nodes", 100)), marker_dim, seed * 2000 + 2
            )

        for delay in args.delays:
            beh = ge.probe_delayed_intact_vs_reset(
                model, datasets["test"], delay=delay, batch_size=32,
                marker_dim=marker_dim, generic_blank=blank,
            )
            g = rg.analyse(model, datasets["test"], delay, marker_dim, blank, seed)
            row = {"sample": sample, "seed": seed,
                   "delayed_intact": float(beh["delayed_intact"]),
                   "delayed_reset": float(beh["delayed_state_reset"]), **g}
            row["geometry_class"] = geometry_class(row)
            rows.append(row)
            print(f"  {sample:>10} seed {seed:>2} delay {delay:>2}: "
                  f"behav={row['delayed_intact']:.3f} decode={row['decode_accuracy']:.3f} "
                  f"cos={row['cos_probe_readout']:.3f} "
                  f"acc0/1={row['readout_accuracy_class0']:.2f}/{row['readout_accuracy_class1']:.2f} "
                  f"bnd={row['boundary_vs_mid']:+.2f} agree={row['prediction_agreement']:.2f} "
                  f"[{row['geometry_class']}]", flush=True)
            pd.DataFrame(rows).to_csv(out_dir / "geometry.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "geometry.csv", index=False)

    # ---------------- group comparison at delay 30 (pre-registered) ----------------
    d30 = df[df.delay == 30].copy()
    d30["behavioural_pass"] = d30["delayed_intact"] >= 0.9
    stats = [
        "decode_accuracy", "cos_probe_readout", "prediction_agreement",
        "boundary_vs_mid", "signed_margin_mean", "abs_margin_mean",
        "state_norm_mean", "class_margin_separation",
        "readout_accuracy", "readout_accuracy_class0", "readout_accuracy_class1",
        "linear_margin_mean",
    ]
    groups: dict = {}
    for label, g in d30.groupby("behavioural_pass"):
        key = "pass" if label else "fail"
        groups[key] = {
            "n": int(len(g)),
            "seeds": sorted(int(s) for s in g.seed),
            **{
                f"{s}_mean": float(g[s].mean()) for s in stats
            },
            **{
                f"{s}_min": float(g[s].min()) for s in stats
            },
            **{
                f"{s}_max": float(g[s].max()) for s in stats
            },
        }

    # ---------------- pre-registered stage-A prediction for stage B ----------------
    # A failing checkpoint is predicted rescuable by readout-only retraining iff
    # the information is present (decode >= 0.95) and the effective readout
    # direction is aligned with it (cos >= 0.90).  The translation statistic
    # explains *why* it currently fails; it does not block rescue.
    failing = d30[~d30.behavioural_pass].copy()
    failing["predicted_rescuable"] = (
        (failing.decode_accuracy >= 0.95) & (failing.cos_probe_readout >= 0.90)
    )
    prediction = {
        "n_failing": int(len(failing)),
        "n_predicted_rescuable": int(failing.predicted_rescuable.sum()),
        "failing_seeds": sorted(int(s) for s in failing.seed),
        "predicted_rescuable_seeds": sorted(
            int(s) for s in failing.loc[failing.predicted_rescuable, "seed"]
        ),
        "n_translated": int((failing.geometry_class == "translated").sum()),
        "n_rotated": int((failing.geometry_class == "rotated").sum()),
        "n_information_absent": int((failing.geometry_class == "information_absent").sum()),
    }

    # consistency with the already-published delay-scaling table (evaluation only)
    consistency = None
    published = ROOT / "results" / "delay_scaling" / "delay_scaling.csv"
    if published.exists():
        prev = pd.read_csv(published)
        prev30 = prev[prev.delay == 30][["sample", "seed", "delayed_intact"]]
        merged = d30.merge(prev30, on=["sample", "seed"], suffixes=("", "_published"))
        consistency = {
            "n": int(len(merged)),
            "max_abs_diff_behaviour": float(
                (merged.delayed_intact - merged.delayed_intact_published).abs().max()
            ),
        }

    summary = {
        "delays": [int(d) for d in args.delays],
        "n_checkpoints": int(df.seed.nunique()),
        "n_rows": int(len(df)),
        "delay30_groups": groups,
        "stageA_prediction": prediction,
        "consistency_with_delay_scaling": consistency,
        "per_checkpoint": {
            f"{r['sample']}_seed{int(r['seed']):03d}_d{int(r['delay'])}": {
                k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in r.items() if k not in ("sample",)
            } for _, r in df.iterrows()
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== delay-30 groups ===")
    for key, g in groups.items():
        print(f"{key:>4} n={g['n']:>2}: cos={g['cos_probe_readout_mean']:.3f} "
              f"agree={g['prediction_agreement_mean']:.3f} "
              f"bnd={g['boundary_vs_mid_mean']:+.2f} "
              f"acc0={g['readout_accuracy_class0_mean']:.2f} "
              f"acc1={g['readout_accuracy_class1_mean']:.2f} "
              f"|margin|={g['abs_margin_mean_mean']:.3f} "
              f"norm={g['state_norm_mean_mean']:.3f}")
    print(f"\nprediction: {prediction['n_predicted_rescuable']}/{prediction['n_failing']} "
          f"failing checkpoints predicted rescuable "
          f"({prediction['n_translated']} translated, {prediction['n_rotated']} rotated, "
          f"{prediction['n_information_absent']} information absent)")
    if consistency:
        print(f"behaviour consistency with delay_scaling.csv: "
              f"max |diff| = {consistency['max_abs_diff_behaviour']:.6f} over {consistency['n']}")
    print(f"\nwrote {out_dir / 'geometry.csv'} and {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
