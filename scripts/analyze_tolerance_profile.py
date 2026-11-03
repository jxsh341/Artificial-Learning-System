#!/usr/bin/env python
"""Score section 8t (tolerance profile) against the pre-registered tests.

Reads results/delay30_tolerance/*__*/tolerance_summary.json + walk_log.csv,
extracts the walk thresholds at the 8q rubric criteria, loads the
interval-censored 8q tolerance observations, and runs P1-P3 + LOO
(seed 14 = named held-out case).  Prints the T1/T2/T3 verdict.

  python scripts/analyze_tolerance_profile.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "delay30_tolerance"

# 8q interval-censored tolerance, d_gnn units (pre-registered table)
TAU = {
    0: (0.0015, 0.0412), 4: (0.0807, None), 6: (0.0558, None), 11: (0.1186, None),
    12: (0.0997, None), 13: (0.0803, None), 14: (None, 0.0012), 16: (0.0523, 0.0811),
    17: (0.0007, 0.0130), 18: (0.0532, None), 21: (0.0400, 0.1149), 22: (0.0071, None),
}


def tau_point(seed: int) -> float:
    lo, hi = TAU[seed]
    if lo is not None and hi is not None:
        return float(np.sqrt(lo * hi))
    if hi is not None:          # censored above (seed 14): midpoint of (0, hi)
        return hi / 2.0
    return lo                    # lower bound only


def threshold(walk: pd.DataFrame, direction: str, col: str, criterion) -> float:
    """Max ladder delta at which the criterion still holds (0 if never;
    np.inf-coded as LADDER top if always)."""
    w = walk[walk.direction == direction].sort_values("delta")
    ok = w[criterion(w)].values
    deltas = w["delta"].values
    if ok.all():
        return float(deltas.max())  # right-censored at ladder top
    good = deltas[ok]
    return float(good.max()) if len(good) else 0.0


def main() -> int:
    rows, walks = [], {}
    for d in sorted(BASE.glob("*__*/tolerance_summary.json")):
        s = json.loads(d.read_text())
        seed = s["seed"]
        walk = pd.read_csv(d.parent / "walk_log.csv")
        walks[seed] = walk
        p1, c = s["part1"], s["curvature"]
        rows.append({
            "seed": seed,
            "tau_lo": TAU[seed][0], "tau_hi": TAU[seed][1], "tau_pt": tau_point(seed),
            "m_behav": p1["m_behav"], "m_decode": p1["m_decode"],
            "disp_state": p1["disp_state"], "class_sep": p1["class_sep_state"],
            "boundary_vs_mid": p1["boundary_vs_mid"], "cos": p1["cos_probe_readout"],
            "state_norm": p1["state_norm_mean"],
            "curv": c["curvature_ratio"], "curv_disp": c["disp_step1"],
            "d_behav_grad": threshold(walk, "u_grad", "delayed_intact",
                                      lambda w: w.delayed_intact >= 1.0),
            "d_decode_grad": threshold(walk, "u_grad", "decode_pre_response",
                                       lambda w: w.decode_pre_response >= 0.95),
            "d_subst_grad": threshold(walk, "u_grad", "subst_diff_follows",
                                      lambda w: w.subst_diff_follows >= 1.0),
            "d_behav_rand": threshold(walk, "u_rand", "delayed_intact",
                                      lambda w: w.delayed_intact >= 1.0),
        })
    t = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    pd.set_option("display.width", 240)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    def sp(x, y, mask=None):
        m = np.ones(len(t), bool) if mask is None else mask
        r = stats.spearmanr(np.asarray(x)[m], np.asarray(y)[m])
        return float(r.statistic), float(r.pvalue)

    tau = t.tau_pt.values
    n11 = (t.seed != 14).values
    print("\n--- P1/P2/P3: Spearman rho vs tau (n=12; n=11 excl. seed 14) ---")
    tests = {
        "m_behav": t.m_behav.values, "m_decode": t.m_decode.values,
        "d_behav_grad": t.d_behav_grad.values, "d_subst_grad": t.d_subst_grad.values,
        "curv (neg)": -t.curv.values, "m_behav/curv": t.m_behav.values / t.curv.values,
    }
    for name, x in tests.items():
        r12, p12 = sp(x, tau)
        r11, p11 = sp(x, tau, n11)
        print(f"  {name:>14}: rho12={r12:+.2f} (p={p12:.3f})   rho11={r11:+.2f} (p={p11:.3f})")

    print("\n--- P2 anisotropy (descriptive) ---")
    ratio = np.where(t.d_behav_rand > 0, t.d_behav_grad / np.maximum(t.d_behav_rand, 1e-9), np.inf)
    for seed, g, r in zip(t.seed, t.d_behav_grad, t.d_behav_rand):
        print(f"  seed {seed:>2}: grad={g:.4g} rand={r:.4g}")

    print("\n--- LOO (log10 tau ~ log10 predictor, fit on 11, predict held-out) ---")
    for name, x in tests.items():
        with np.errstate(divide="ignore"):
            lx = np.log10(np.maximum(np.asarray(x, float), 1e-9))
        ly = np.log10(np.maximum(tau, 1e-9))
        errs, pred14 = [], None
        for i in range(len(t)):
            m = np.ones(len(t), bool)
            m[i] = False
            b, a = np.polyfit(lx[m], ly[m], 1)  # slope, intercept
            yhat = a + b * lx[i]
            errs.append(abs(yhat - ly[i]))
            if t.seed[i] == 14:
                pred14 = 10 ** yhat
        print(f"  {name:>14}: LOO median |log10 err|={np.median(errs):.2f}  "
              f"seed14 predicted tau={pred14:.2e} (observed < 0.0012)")

    p1_ok = any(sp(x, tau, n11)[0] >= 0.6 for x in
                (t.m_behav.values, t.m_decode.values, t.d_behav_grad.values))
    print("\n--- verdict ---")
    print(f"P1/P2 rank criterion (rho >= 0.6 on n=11): {p1_ok}")
    print("registered T1 requires: rank criterion AND LOO median |log10 err| <= 1.0 "
          "AND seed14 LOO tau < 0.01")
    t.to_csv(BASE / "tolerance_table.csv", index=False)
    print(f"wrote {BASE / 'tolerance_table.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
