#!/usr/bin/env python
"""Stage B of §8l: H7b at delay 30, analysed against the stage-A geometry.

Reads the completed intervention runs and joins them, per checkpoint, to the
pre-intervention geometry measured in stage A.  Classification is by the full
pre-registered rubric, never by accuracy alone.

  python scripts/analyze_delay30_rescue.py \
      [--results-dir results/delay30_h7b] \
      [--geometry results/readout_geometry/geometry.csv]

Outputs results/delay30_h7b/per_checkpoint.csv and analysis.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _sample_of(path: str) -> str:
    return "population" if "d800_population" in (path or "") else "original"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/delay30_h7b")
    ap.add_argument("--geometry", default="results/readout_geometry/geometry.csv")
    args = ap.parse_args()

    out_dir = ROOT / args.results_dir
    summary = json.loads((out_dir / "summary.json").read_text())
    geo = pd.read_csv(ROOT / args.geometry)
    geo30 = geo[geo.delay == 30].copy()

    rows: list[dict] = []
    for key, s in summary.items():
        sample = _sample_of(s.get("source_checkpoint", ""))
        seed = int(s["seed"])
        g = geo30[(geo30["sample"] == sample) & (geo30["seed"] == seed)]
        grow = g.iloc[0] if len(g) else None
        r = s["rubric"]
        b = s["battery"]
        pre, post = s["pre_analysis"], s["post_analysis"]
        cap = s.get("capture_info")
        row = {
            "run": key,
            "sample": sample,
            "seed": seed,
            "capture": bool(cap is not None),
            "capture_episode": int(cap["episode_in_stage"]) if cap else -1,
            "capture_delayed_at_capture": float(cap["delayed_intact"]) if cap else float("nan"),
            "immediate": float(b["immediate"]),
            "delayed_intact": float(b["delayed_intact"]),
            "delayed_reset": float(b["delayed_state_reset"]),
            "subst_same": float(post.get("subst_same_answer_accuracy", float("nan"))),
            "subst_diff_follows": float(r["subst_diff_follows"]),
            "subst_zero": float(post.get("subst_zero_accuracy", float("nan"))),
            "subst_shuffled": float(post.get("subst_shuffled_accuracy", float("nan"))),
            "decode_pre": float(pre.get("decode_pre_response", float("nan"))),
            "decode_post": float(post.get("decode_pre_response", float("nan"))),
            "d_head_stage4": float(s["param_deltas_stage4"]["head"]),
            "d_C_stage4": float(s["param_deltas_stage4"]["ssm_C"]),
            "d_D_stage4": float(s["param_deltas_stage4"]["ssm_D"]),
            "d_gnn_stage4": float(s["param_deltas_stage4"]["gnn"]),
            "d_pool_stage4": float(s["param_deltas_stage4"]["pooling"]),
            "d_gen_stage4": float(s["param_deltas_stage4"]["ssm_gen"]),
            "all_criteria": bool(r["all_criteria"]),
        }
        if grow is not None:
            row.update({
                "geo_cos": float(grow.cos_probe_readout),
                "geo_agreement": float(grow.prediction_agreement),
                "geo_boundary_vs_mid": float(grow.boundary_vs_mid),
                "geo_abs_margin": float(grow.abs_margin_mean),
                "geo_signed_margin": float(grow.signed_margin_mean),
                "geo_state_norm": float(grow.state_norm_mean),
                "geo_readout_acc": float(grow.readout_accuracy),
                "geo_class0_acc": float(grow.readout_accuracy_class0),
                "geo_class1_acc": float(grow.readout_accuracy_class1),
                "geo_class": str(grow.geometry_class),
            })
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["sample", "seed"]).reset_index(drop=True)
    df.to_csv(out_dir / "per_checkpoint.csv", index=False)

    rescued = df[df.all_criteria]
    not_rescued = df[~df.all_criteria]
    captured = df[df.capture]

    # geometry comparison (only meaningful if the outcome is partial)
    geo_stats = ["geo_cos", "geo_agreement", "geo_boundary_vs_mid",
                 "geo_abs_margin", "geo_signed_margin", "geo_state_norm"]
    geo_compare: dict = {}
    for stat in geo_stats:
        if stat not in df.columns:
            continue
        a = rescued[stat].dropna()
        b = not_rescued[stat].dropna()
        geo_compare[stat] = {
            "rescued_mean": float(a.mean()) if len(a) else None,
            "not_rescued_mean": float(b.mean()) if len(b) else None,
            "rescued_values": [float(x) for x in a],
            "not_rescued_values": [float(x) for x in b],
            "rank_separated": bool(len(a) and len(b) and
                                   (a.max() < b.min() or b.max() < a.min())),
        }

    analysis = {
        "n": int(len(df)),
        "n_rescued": int(len(rescued)),
        "n_not_rescued": int(len(not_rescued)),
        "n_captured": int(len(captured)),
        "rescued_seeds": [int(x) for x in rescued.seed],
        "not_rescued_seeds": [int(x) for x in not_rescued.seed],
        "capture_never_fired_seeds": [int(x) for x in df[~df.capture].seed],
        "all_criteria_rule": (
            "immediate>=0.90 and delayed>=0.90 and same>=0.90 and "
            "follows>=0.90 and zero<=0.65 and shuffled<=0.65 and decode>=0.95"
        ),
        "geometry_comparison": geo_compare,
    }
    (out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2, default=float))

    print(f"{'run':<34} {'cap':>4} {'imm':>5} {'del':>5} {'same':>5} {'follow':>6} "
          f"{'zero':>5} {'shuf':>5} {'dec':>5} {'cos':>5} {'agree':>5} {'bnd':>6}  rubric")
    for r in df.itertuples():
        print(f"{r.run:<34} {('ep'+str(r.capture_episode)) if r.capture else 'NONE':>4} "
              f"{r.immediate:5.3f} {r.delayed_intact:5.3f} {r.subst_same:5.3f} "
              f"{r.subst_diff_follows:6.3f} {r.subst_zero:5.3f} {r.subst_shuffled:5.3f} "
              f"{r.decode_post:5.3f} {r.geo_cos:5.3f} {r.geo_agreement:5.3f} "
              f"{r.geo_boundary_vs_mid:+6.2f}  {'PASS' if r.all_criteria else 'fail'}")
    print(f"\ncaptured {analysis['n_captured']}/{analysis['n']}; "
          f"full rubric {analysis['n_rescued']}/{analysis['n']}")
    if 0 < analysis["n_rescued"] < analysis["n"]:
        print("geometry, rescued vs not (rank separation):")
        for stat, v in geo_compare.items():
            print(f"  {stat:<22} rescued={v['rescued_mean']}  not={v['not_rescued_mean']}  "
                  f"separated={v['rank_separated']}")
    print(f"\nwrote {out_dir / 'per_checkpoint.csv'} and {out_dir / 'analysis.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
