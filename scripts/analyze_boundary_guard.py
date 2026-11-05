#!/usr/bin/env python
"""Section 8v V3 scorer: bvm-guarded arm vs the fixed eps=0.2 arm.

Registered outcomes (registration + amendments):
  V3-E1 (primary): guarded runs starting access-ON have ZERO high->low
        behavioural transitions (target >= 10 of eligible checkpoints);
        per-run counts reported.
  V3-E2: decode (ridge CV, the canonical instrument) = 1.000 at every episode
        of every guarded run (any drop recorded as measured).
  V3-E3: full-rubric pass count >= the fixed arm's (fresh: same environment;
        recorded: 7/12 in section 8q); and on each checkpoint that adapted in
        the fixed arm, guarded GNN delta >= 50% of fixed-arm GNN delta.
  V3-E4 (measured): guard activity (reject counts, consecutive-reject streaks,
        rejected-step fraction), adaptation deltas, arm-vs-arm bvm stats.
  V3-E5 tiers as registered.

  python scripts/analyze_boundary_guard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FRAG = Path("results/delay30_fragility")
GUARD = Path("results/delay30_fragility_guard")
OSC_SEEDS = {0, 4, 6, 11, 12, 13, 16, 17, 21}
STABLE_SEEDS = {14, 18, 22}


def transitions(delayed: np.ndarray) -> list[int]:
    d = np.asarray(delayed, dtype=float)
    out = []
    for e in range(1, len(d)):
        if d[e] < 0.6 and d[e - 1] >= 0.9:
            out.append(e)
            for j in range(e, len(d)):
                if d[j] >= 0.9:
                    e = j
                    break
            else:
                break
    return out


def load(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(run_dir / "per_episode.csv")
    s = json.loads((run_dir / "arm_summary.json").read_text())
    return df, s


def main() -> int:
    # ---- control (fixed eps=0.2, fresh environment, registered recipe) ----
    ctrl = {}
    for d in sorted(FRAG.glob("*__*/arm_summary.json")):
        seed = int(d.parent.name.split("seed")[-1][:3])
        df, s = load(d.parent)
        ctrl[seed] = {"df": df, "summary": s,
                      "trans": transitions(df["delayed_intact"].to_numpy(float)),
                      "gnn": s["param_deltas"]["gnn"],
                      "rubric": s["rubric"]}
    guard = {}
    for d in sorted(GUARD.glob("*__*/arm_summary.json")):
        seed = int(d.parent.name.split("seed")[-1][:3])
        df, s = load(d.parent)
        guard[seed] = {"df": df, "summary": s,
                       "trans": transitions(df["delayed_intact"].to_numpy(float)),
                       "gnn": s["param_deltas"]["gnn"],
                       "rubric": s["rubric"],
                       "guard": s.get("guard_stats", {})}

    if not guard:
        print("no guard runs found")
        return 1

    pd.set_option("display.width", 240)
    rows = []
    for seed in sorted(guard):
        g, c = guard[seed], ctrl.get(seed)
        n_guard_trans = len(g["trans"])
        n_ctrl_trans = len(c["trans"]) if c else None
        dec = g["df"].get("decode_pre_response_cv")
        dec_min = float(dec.min()) if dec is not None else float("nan")
        dec_below = int((dec < 0.995).sum()) if dec is not None else 0
        bvm0 = float(g["df"]["boundary_vs_mid"].iloc[0])
        starts_on = (abs(bvm0) < 0.5)
        rej = g["guard"].get("total_rejects"); streak = g["guard"].get("max_consecutive_reject_episodes")
        # E4 fidelity: accepted-step bvm_pred vs next episode's actual test bvm
        sl = GUARD / f"D800_interleaved_seed{seed:03d}__guard" / "step_log.csv"
        err_mean = float("nan")
        if sl.exists():
            sl_df = pd.read_csv(sl)
            if "bvm_pred" in sl_df.columns:
                pred = sl_df["bvm_pred"].to_numpy(float)
                actual = g["df"]["boundary_vs_mid"].to_numpy(float)
                n = min(len(pred), len(actual) - 1)
                ok = pred[:n] == pred[:n]
                if ok.any():
                    err_mean = float(np.mean(np.abs(pred[:n][ok] - actual[1:1 + n][ok])))
        gnn_ratio = (g["gnn"] / c["gnn"]) if (c and c["gnn"] > 0) else float("nan")
        rows.append({
            "seed": seed, "starts_on": starts_on,
            "guard_trans": n_guard_trans, "ctrl_trans": n_ctrl_trans,
            "decode_min": dec_min, "decode_below_0.995": dec_below,
            "gnn_guard": g["gnn"], "gnn_ctrl": (c["gnn"] if c else np.nan),
            "gnn_ratio": gnn_ratio,
            "pred_err_mean": err_mean,
            "rubric_guard_delayed": g["rubric"]["delayed_ok"],
            "guard_rejects": rej, "max_streak": streak,
        })
    t = pd.DataFrame(rows)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- E1 (registered letter: zero crossings for >= 10 of eligible) ----
    eligible = t[t.starts_on]
    zero = int((eligible.guard_trans == 0).sum())
    e1 = "PASS" if len(eligible) and zero >= 10 else \
         "TIER-partial" if zero >= 5 else "FAIL"
    print(f"\nV3-E1: {zero}/{len(eligible)} eligible starts with zero crossings "
          f"-> {e1}")
    if (eligible.guard_trans > 0).any():
        bad = eligible[eligible.guard_trans > 0]
        print("  runs with crossings:", bad[["seed", "guard_trans", "ctrl_trans"]].to_string(index=False))

    # ---- E2 ----
    tot_below = int(t["decode_below_0.995"].sum())
    e2 = "PASS" if tot_below == 0 else f"MEASURED ({tot_below} episodes below 0.995)"
    print(f"V3-E2: decode violations {tot_below} -> {e2}")

    # ---- E3 ----
    g_pass = int(sum(bool(r["rubric_guard_delayed"]) for r in rows))
    c_pass = int(sum(bool(r["rubric"]["delayed_ok"]) for r in ctrl.values()))
    adapted = [r for r in rows if r["gnn_ctrl"] > 0.01]
    ratio_bad = [r["seed"] for r in adapted if r["gnn_ratio"] < 0.5]
    e3 = ("PASS" if g_pass >= 7 and not ratio_bad else
          "TIER-partial" if not ratio_bad else "FAIL")
    print(f"V3-E3: rubric guard {g_pass}/12 vs fresh control {c_pass}/12 "
          f"(recorded 8q: 7/12); adapted checkpoints with <50% of control GNN "
          f"delta: {ratio_bad} ({len(ratio_bad)}/{len(adapted)} adapted) -> {e3}")

    # ---- E4 ----
    print("\nV3-E4 (measured): guard rejects %d total, max consecutive streak %s, "
          "mean accepted-frac per run %.3f, mean prediction error %.2f bvm units"
          % (sum(r["guard_rejects"] or 0 for r in rows),
             max((r["max_streak"] or 0) for r in rows),
             np.mean([1 - (guard[r["seed"]]["guard"].get("rejected_steps_frac") or 0.0)
                      for r in rows]),
             np.nanmean(t["pred_err_mean"])))

    (GUARD / "analysis_guard.json").write_text(
        json.dumps({r["seed"]: r for r in rows}, indent=2, default=float))
    print(f"\nwrote {GUARD / 'analysis_guard.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
