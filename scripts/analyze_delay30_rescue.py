#!/usr/bin/env python
"""Stage B of §8l: H7b at delay 30, analysed against the stage-A geometry.

Reads the completed intervention runs and joins them, per checkpoint, to the
pre-intervention geometry measured in stage A.  The pre-registered success
criterion is the *full* rubric after the joint stage; the stage-3 (readout-only)
outcome is reported separately as a diagnostic, so a rescue that is later
erased by the joint stage cannot be mistaken for a full success.

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

import numpy as np
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
        # stage-3 (readout-only) outcome, before the joint stage
        stage3_delayed = float("nan")
        stage3_restore_delayed = float("nan")
        probes_path = out_dir / key / "training_probes.csv"
        if probes_path.exists():
            probes = pd.read_csv(probes_path)
            row3 = probes[probes.phase == "post_stage"]
            row4 = probes[probes.phase == "post_restore"]
            if len(row3):
                stage3_delayed = float(row3.iloc[0].delayed_intact)
            if len(row4):
                stage3_restore_delayed = float(row4.iloc[0].delayed_intact)
        stage3 = s.get("stage3_analysis", {})
        stage3_decode = float(stage3.get("decode_pre_response", float("nan")))
        stage3_follows = float(stage3.get("subst_diff_follows", float("nan")))
        stage3_rescued = bool(
            np.isfinite(stage3_delayed) and stage3_delayed >= 0.9
            and stage3_decode >= 0.95 and stage3_follows >= 0.9
        )
        row = {
            "run": key,
            "sample": sample,
            "seed": seed,
            "capture": bool(cap is not None),
            "capture_episode": int(cap["episode_in_stage"]) if cap else -1,
            "capture_delayed_at_capture": float(cap["delayed_intact"]) if cap else float("nan"),
            "stage3_delayed": stage3_delayed,
            "stage3_restore_delayed": stage3_restore_delayed,
            "stage3_decode": stage3_decode,
            "stage3_follows": stage3_follows,
            "stage3_rescued": stage3_rescued,
            "immediate": float(b["immediate"]),
            "delayed_intact": float(b["delayed_intact"]),
            "delayed_reset": float(b["delayed_reset"]),
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
        "n_stage3_rescued": int(df.stage3_rescued.sum()),
        "n_eroded_by_stage4": int((df.stage3_rescued & ~df.all_criteria).sum()),
        "rescued_seeds": [int(x) for x in rescued.seed],
        "not_rescued_seeds": [int(x) for x in not_rescued.seed],
        "stage3_rescued_seeds": [int(x) for x in df[df.stage3_rescued].seed],
        "capture_never_fired_seeds": [int(x) for x in df[~df.capture].seed],
        "all_criteria_rule": (
            "immediate>=0.90 and delayed>=0.90 and same>=0.90 and "
            "follows>=0.90 and zero<=0.65 and shuffled<=0.65 and decode>=0.95"
        ),
        "stage3_rule": (
            "post-stage-3 delayed>=0.90 and stage-3 decode>=0.95 and "
            "stage-3 substitution follows>=0.90"
        ),
        "geometry_comparison": geo_compare,
    }
    (out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2, default=float))

    print(f"{'run':<34} {'cap':>4} {'ep':>3} {'s3del':>6} {'s3dec':>6} "
          f"{'findel':>6} {'findec':>6} {'follow':>6}  rubric")
    for r in df.itertuples():
        print(f"{r.run:<34} {('yes' if r.capture else 'NONE'):>4} {r.capture_episode:>3} "
              f"{r.stage3_delayed:6.3f} {r.stage3_decode:6.3f} "
              f"{r.delayed_intact:6.3f} {r.decode_post:6.3f} {r.subst_diff_follows:6.3f}  "
              f"{'PASS' if r.all_criteria else 'fail'}")
    print(f"\ncaptured {analysis['n_captured']}/{analysis['n']}; "
          f"stage-3 (readout-only) rescued {analysis['n_stage3_rescued']}/{analysis['n']}; "
          f"full rubric {analysis['n_rescued']}/{analysis['n']}; "
          f"eroded by stage 4 {analysis['n_eroded_by_stage4']}/{analysis['n']}")
    if 0 < analysis["n_rescued"] < analysis["n"]:
        print("geometry, rescued vs not (rank separation):")
        for stat, v in geo_compare.items():
            print(f"  {stat:<22} rescued={v['rescued_mean']}  not={v['not_rescued_mean']}  "
                  f"separated={v['rank_separated']}")
    print(f"\nwrote {out_dir / 'per_checkpoint.csv'} and {out_dir / 'analysis.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
