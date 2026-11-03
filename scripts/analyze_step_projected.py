#!/usr/bin/env python
"""Score section 8s (post-Adam step projection) against the pre-registered grid.

Reads results/delay30_step_proj/*__*/arm_summary.json (plus training_probes.csv
for the oscillation count), assigns each checkpoint its first-match outcome
category, and prints the verdict.

Categories (first match wins, as pre-registered):
  1 full_success       rubric PASS, delayed 1.000, immediate 1.000, GNN delta >= 0.5,
                       < 3 hi/lo transitions in the delayed trajectory
  2 partial_success    decode >= 0.95 and >= 3 transitions
  3 boundary_translation decode >= 0.95, follows < 0.9, < 3 transitions
  4 representation_collapse decode < 0.95
  5 no_adaptation      GNN delta < 0.05 and category 1 not met

Verdict: confirmed iff >= 8/12 category 1; partial iff >= 8/12 categories 1-2;
rejected iff >= 8/12 categories 3-4; category-5-dominant reported separately as
"no admissible direction survives magnitude restoration".

  python scripts/analyze_step_projected.py [--results-dir results/delay30_step_proj]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def transitions(values) -> int:
    """Hysteresis count between >= 0.9 (hi) and <= 0.6 (lo), as registered."""
    n, state = 0, None
    for x in values:
        cur = "hi" if x >= 0.9 else ("lo" if x <= 0.6 else None)
        if cur is not None and state is not None and cur != state:
            n += 1
        if cur is not None:
            state = cur
    return n


def categorize(s: dict, n_trans: int) -> str:
    r, b = s["rubric"], s["battery"]
    decode = s["post_analysis"].get("decode_pre_response", float("nan"))
    d_gnn = s["param_deltas"]["gnn"]
    delayed = b["delayed_intact"]
    immediate = b.get("immediate", float("nan"))
    follows = r["subst_diff_follows"]

    if (r["all_criteria"] and delayed >= 1.0 and immediate >= 1.0
            and d_gnn >= 0.5 and n_trans < 3):
        return "1_full_success"
    if decode >= 0.95 and n_trans >= 3:
        return "2_partial_success"
    if decode >= 0.95 and follows < 0.9 and n_trans < 3:
        return "3_boundary_translation"
    if decode < 0.95:
        return "4_representation_collapse"
    if d_gnn < 0.05:
        return "5_no_adaptation"
    return "5_no_adaptation"  # residual: no positive evidence of success


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/delay30_step_proj")
    args = ap.parse_args()
    base = ROOT / args.results_dir

    rows = []
    for f in sorted(base.glob("*__*/arm_summary.json")):
        s = json.loads(f.read_text())
        tr = pd.read_csv(f.parent / "training_probes.csv")
        beh = tr[tr.phase != "pre"].delayed_intact.values
        nt = transitions(beh)
        sl = pd.read_csv(f.parent / "step_log.csv")
        rows.append({
            "seed": s["seed"],
            "cat": categorize(s, nt),
            "delayed": s["battery"]["delayed_intact"],
            "immediate": s["battery"].get("immediate"),
            "follows": s["rubric"]["subst_diff_follows"],
            "same_ok": s["rubric"]["subst_same_ok"],
            "decode": s["post_analysis"].get("decode_pre_response"),
            "d_gnn": s["param_deltas"]["gnn"],
            "d_pool": s["param_deltas"]["pooling"],
            "d_gen": s["param_deltas"]["ssm_gen"],
            "ratio_mean": s["projection_ratio_mean"],
            "ratio_min": s["projection_ratio_min"],
            "adam_norm": s["adam_update_norm_mean"],
            "disp_mean": s["displacement_mean"],
            "disp_max": s["displacement_max"],
            "disp_final": s["displacement_final"],
            "rej": s["rejected_steps"],
            "trans": nt,
            "traj": " ".join(f"{x:.2f}" for x in beh[:15]),
        })

    t = pd.DataFrame(rows).sort_values("seed")
    pd.set_option("display.width", 260)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    counts = t["cat"].value_counts().to_dict()
    print("\ncategory counts:", counts)
    n1 = counts.get("1_full_success", 0)
    n12 = n1 + counts.get("2_partial_success", 0)
    n34 = counts.get("3_boundary_translation", 0) + counts.get("4_representation_collapse", 0)
    n5 = counts.get("5_no_adaptation", 0)
    if n1 >= 8:
        verdict = "CONFIRMED (>= 8/12 category 1)"
    elif n12 >= 8:
        verdict = "PARTIAL SUPPORT (>= 8/12 categories 1-2)"
    elif n34 >= 8:
        verdict = "REJECTED (>= 8/12 categories 3-4)"
    else:
        verdict = "mixed / category-5 dominant"
    print(f"verdict: {verdict}")
    if n5:
        print(f"note: {n5}/12 have no admissible direction surviving restoration")

    s14 = t[t["seed"] == 14]
    if len(s14):
        r = s14.iloc[0]
        print(f"\nseed 14 (registered diagnostic): {r['cat']}  decode={r['decode']:.3f} "
              f"follows={r['follows']:.3f} d_gnn={r['d_gnn']:.3f} "
              f"ratio={r['ratio_mean']:.3f} rej={r['rej']}")
    out = base / "categories.csv"
    t.to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
