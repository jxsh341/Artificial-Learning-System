#!/usr/bin/env python
"""Section 8v scorer: online fragility indicator (V1 detection, V2 prediction).

Registered rules (amendment 1, |bvm| form):
  V1-D: access-ON iff |bvm| < 0.5, OFF iff |bvm| >= 0.5, per episode; observed
        phase from the run's own quick probes (delayed >= 0.6 = ON-side).
        Zero violations across all 12 runs = registered success (tiered
        otherwise: >=99%, >=97%, <97% of episodes).
  V1-C: seed 14 classified OFF continuously; violations on 14/18/22 <= 1
        episode total.
  V2-P: fire at episode e iff |bvm_e| + v_e >= EDGE (=0.4948) with
        v_e = |bvm_e| - |bvm_{e-1}| >= 0 (toward-edge velocity).  A fire is a
        forecast for a high->low drop in (e, e+3]; drop = first episode with
        delayed < 0.6 whose predecessor is >= 0.9.  Sensitivity = fraction of
        transitions with >=1 fire in the 3-episode pre-window (target >= 90%);
        false-alarm rate <= 10%.
  V2-N: fires on seeds 14/18/22 <= 1 per run (the null guards).
  V2-B: oracle-swept baselines under identical accounting --
        (a) cum_displacement level threshold, (b) |bvm_e| + |v_e| >= tau
        (no direction condition).  Registered claim fired only if the
        bvm+velocity rule beats the best oracle threshold of both.

  python scripts/analyze_online_fragility.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("results/delay30_fragility")
EDGE = 0.4948          # measured access-ON edge (amendment 1); in-sample constant
T_RULE = 0.5           # pinned V1 threshold (mid-gap)
WINDOW = 3             # false-alarm horizon (episodes), registered
OSC_SEEDS = {0, 4, 6, 11, 12, 13, 16, 17, 21}
STABLE_SEEDS = {14, 18, 22}

TAU_GRID = None  # built per cohort from quantiles of the scored streams


def transitions(delayed: np.ndarray) -> list[int]:
    """First episode with delayed < 0.6 whose predecessor is >= 0.9 (hi->lo)."""
    d = np.asarray(delayed, dtype=float)
    out = []
    for e in range(1, len(d)):
        if d[e] < 0.6 and d[e - 1] >= 0.9:
            out.append(e)
            for j in range(e, len(d)):   # skip ahead until re-entry >= 0.9
                if d[j] >= 0.9:
                    e = j  # noqa
                    break
            else:
                break
    return out


def fires_rule(bvm: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    """Amendment-2 gated rule (registered intent): |bvm_e| < 0.5 and v_e >= 0
    and |bvm_e| + v_e >= EDGE.  Returns (fire, no_gate) — the no-gate stream
    is the amendment-1 letter, scored alongside."""
    b = np.abs(np.asarray(bvm, dtype=float))
    v = np.concatenate([[np.nan], np.diff(b)])
    base = (b + v >= EDGE) & (v >= 0) & (b == b)
    return base & (b < 0.5), base


def drop_episodes(delayed: np.ndarray) -> set[int]:
    """All episodes in a low phase (delayed < 0.6 OR belonging to the drop
    segment).  Used for false-alarm accounting."""
    d = np.asarray(delayed, dtype=float)
    low = np.zeros(len(d), dtype=bool)
    for e in range(1, len(d)):
        if d[e] < 0.6:
            # low phase: extend until a >= 0.9 re-entry
            j = e
            while j < len(d) and d[j] < 0.9:
                low[j] = True
                j += 1
            e = j
    return set(np.flatnonzero(low))


def account(fire: np.ndarray, drops: set[int], trans: list[int]) -> dict:
    """Identical accounting for any firing stream."""
    fire_idx = list(np.flatnonzero(fire))
    tp, fa = [], []
    for e in fire_idx:
        hit = next((t for t in trans if e < t <= e + WINDOW), None)
        (tp if hit is not None else fa).append(e)
    covered = {t for t in trans if any(e < t <= e + WINDOW for e in fire_idx)}
    n_fire = len(fire_idx)
    return {
        "n_fires": n_fire, "tp_fires": tp, "fa_fires": fa,
        "n_transitions": len(trans), "transitions": trans,
        "covered_transitions": sorted(covered),
        "sensitivity": (len(covered) / len(trans)) if trans else float("nan"),
        "fa_rate": (len(fa) / n_fire) if n_fire else 0.0,
    }


def oracle_baseline_b(df: pd.DataFrame) -> dict:
    """(b) |bvm_e| + |v_e| >= tau, no direction condition; tau swept (oracle).
    Primary scoring restricts fires to the ON side (|bvm| < 0.5), matching the
    V2 gate; the unrestricted form is returned alongside."""
    b = np.abs(df["boundary_vs_mid"].to_numpy(dtype=float))
    v = np.concatenate([[np.nan], np.diff(b)])
    delayed = df["delayed_intact"].to_numpy(dtype=float)
    trans = transitions(delayed)
    drops = drop_episodes(delayed)
    best = best_u = None
    for tau in np.arange(0.30, 0.75, 0.01):
        fire_u = (b + v >= tau) & (b == b)
        fire = fire_u & (b < 0.5)
        if fire.any():
            r = account(fire, drops, trans)
            score = (r["sensitivity"] - 2 * r["fa_rate"])
            if best is None or score > best[0]:
                best = (score, tau, r)
        if fire_u.any():
            ru = account(fire_u, drops, trans)
            score_u = (ru["sensitivity"] - 2 * ru["fa_rate"])
            if best_u is None or score_u > best_u[0]:
                best_u = (score_u, tau, ru)
    if best is None:
        return {"n_fires": 0, "sensitivity": float("nan"), "fa_rate": float("nan")}
    out = best[2] | {"best_tau": float(best[1])}
    if best_u is not None:
        out["unrestricted"] = best_u[2] | {"best_tau": float(best_u[1])}
    return out


def oracle_baseline_a(df: pd.DataFrame) -> dict:
    """(a) cum_displacement level threshold; swept (oracle)."""
    c = df["cum_displacement"].to_numpy(dtype=float)
    delayed = df["delayed_intact"].to_numpy(dtype=float)
    trans = transitions(delayed)
    drops = drop_episodes(delayed)
    best = None
    for tau in np.arange(0.02, 0.22, 0.005):
        fire = np.zeros(len(c), dtype=bool)
        for e in range(1, len(c)):
            if c[e - 1] < tau <= c[e]:
                fire[e] = True
        if fire.any():
            r = account(fire, drops, trans)
            score = (r["sensitivity"] - 2 * r["fa_rate"])
            if best is None or score > best[0]:
                best = (score, tau, r)
    if best is None:
        return {"n_fires": 0, "sensitivity": float("nan"), "fa_rate": float("nan")}
    return best[2] | {"best_tau": float(best[1])}


def v1_score(df: pd.DataFrame) -> dict:
    bvm = df["boundary_vs_mid"].to_numpy(dtype=float)
    delayed = df["delayed_intact"].to_numpy(dtype=float)
    ok = bvm == bvm
    pred_on = np.abs(bvm) < T_RULE
    obs_on = delayed >= 0.6
    viol = (pred_on != obs_on) & ok
    return {"n_episodes": int(ok.sum()), "violations": int(viol.sum()),
            "violation_episodes": [int(df.episode[i]) for i in np.flatnonzero(viol)],
            "frac_correct": float(1 - viol.sum() / max(int(ok.sum()), 1))}


def main() -> int:
    runs = []
    for d in sorted(BASE.glob("*__*/arm_summary.json")):
        run_dir = d.parent
        df = pd.read_csv(run_dir / "per_episode.csv")
        seed = int(run_dir.name.split("seed")[-1][:3])
        delayed = df["delayed_intact"].to_numpy(dtype=float)
        trans = transitions(delayed)
        drops = drop_episodes(delayed)

        fire, fire_l = fires_rule(df["boundary_vs_mid"].to_numpy(dtype=float))
        acc = account(fire, drops, trans)
        acc_l = account(fire_l, drops, trans)

        out = {
            "seed": seed,
            "q_check": json.loads(d.read_text())["q_determinism_check"],
            "v1": v1_score(df),
            "v2": acc,
            "v2_letter": acc_l,
            "v2_n_fires": int(acc["n_fires"]),
            "v2_letter_n_fires": int(acc_l["n_fires"]),
            "v2_baseline_a": oracle_baseline_a(df),
            "v2_baseline_b": oracle_baseline_b(df),
            "n_transitions": len(trans),
        }
        # seed-level guard for the stable comparators
        if seed in STABLE_SEEDS:
            out["v2_n"] = {"fires": int(acc["n_fires"])}
        runs.append(out)

    if not runs:
        print("no runs found")
        return 1

    pd.set_option("display.width", 220)
    t = pd.DataFrame([{k: r[k] for k in ("seed", "n_transitions", "v2_n_fires")}
                      for r in runs])
    print(t.to_string(index=False))

    print("\n--- V1-D ---")
    tot_eps = tot_viol = 0
    for r in runs:
        v = r["v1"]
        tot_eps += v["n_episodes"]
        tot_viol += v["violations"]
        print(f"seed {r['seed']:>2}: violations {v['violations']}/{v['n_episodes']} "
              f"({v['violation_episodes'][:6]}{'...' if len(v['violation_episodes']) > 6 else ''})")
    frac = 1 - tot_viol / max(tot_eps, 1)
    v1_verdict = ("PASS" if tot_viol == 0 else
                  "TIER-1" if frac >= 0.99 else "TIER-2" if frac >= 0.97 else "FAIL")
    print(f"V1-D: {tot_viol} violations / {tot_eps} episodes -> {v1_verdict}")

    print("\n--- V2-P / V2-N (gated registered-intent rule) ---")
    osc = [r for r in runs if r["seed"] in OSC_SEEDS]
    stables = [r for r in runs if r["seed"] in STABLE_SEEDS]

    def cohort_v2(key):
        n_trans = sum(r[key]["n_transitions"] for r in osc)
        n_cov = sum(len(r[key]["covered_transitions"]) for r in osc)
        n_fire = sum(r[key]["n_fires"] for r in osc)
        n_fa = sum(len(r[key]["fa_fires"]) for r in osc)
        sens = n_cov / n_trans if n_trans else float("nan")
        fa = n_fa / n_fire if n_fire else 0.0
        return n_trans, n_cov, sens, n_fa, n_fire, fa

    for label, key in (("gated (registered intent)", "v2"),
                       ("no-gate (amendment-1 letter)", "v2_letter")):
        n_trans, n_cov, sens, n_fa, n_fire, fa = cohort_v2(key)
        verdict = ("PASS" if sens > 0.90 and fa <= 0.10 else
                   "TIER-1" if sens > 0.75 and fa <= 0.10 else
                   "TIER-2" if sens > 0.50 and fa <= 0.10 else "FAIL")
        print(f"V2-P [{label}]: sens {n_cov}/{n_trans} = {sens:.3f} "
              f"FA {n_fa}/{n_fire} = {fa:.3f} -> {verdict}")
    for r in runs:
        leads = []
        for e in r["v2"]["tp_fires"]:
            t2 = next((x for x in r["v2"]["transitions"] if e < x <= e + WINDOW), None)
            if t2 is not None:
                leads.append(t2 - e)
        print(f"seed {r['seed']:>2}: trans={r['v2']['n_transitions']} "
              f"gated: fires={r['v2']['n_fires']} cov={len(r['v2']['covered_transitions'])} "
              f"fa={len(r['v2']['fa_fires'])} leads={leads[:6]} | "
              f"letter fires={r['v2_letter_n_fires']} fa={len(r['v2_letter']['fa_fires'])}")
    for r in stables:
        print(f"V2-N seed {r['seed']:>2}: fires {r['v2_n_fires']} "
              f"({'OK' if r['v2_n_fires'] <= 1 else 'VIOLATION'})")

    print("\n--- V2-B (oracle baselines, cohort-pooled accounting) ---")
    for name, key in (("a: cum_disp", "v2_baseline_a"), ("b: |bvm|+|v|", "v2_baseline_b")):
        sensb = [r[key]["sensitivity"] for r in osc if r[key]["sensitivity"] == r[key]["sensitivity"]]
        fab = [r[key]["fa_rate"] for r in osc]
        print(f"{name}: mean sensitivity {np.nanmean(sensb):.3f}, mean FA {np.mean(fab):.3f}")

    (BASE / "analysis_fragility.json").write_text(
        json.dumps(runs, indent=2, default=float))
    print(f"\nwrote {BASE / 'analysis_fragility.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
