#!/usr/bin/env python
"""Score section 8u (access-oscillation mechanism) against the pre-registered
mechanism outcomes M1-M4.

Reads results/delay30_oscillation/*__*/{per_episode.csv, dense_probes.csv,
j_instruments.csv, dense_dirs.npz, arm_summary.json} and produces:
- per-run phenotype classification (behavioural / functional / representational
  oscillation — registered as non-interchangeable);
- phase segmentation and instrumental ordering at each high->low crossing;
- U2 interrupt report (battery, no-update repeats, frozen-hold, resume check);
- U3 cycle-comparative geometry (same representation vs replacement);
- M1/M2/M3/M4 verdicts per oscillator and for the population.

  python scripts/analyze_oscillation_mechanism.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "delay30_oscillation"
HI, LO = 0.9, 0.6        # registered hysteresis
DECODE_OK = 0.95         # registered decode criterion
MARGIN_DROP = 0.5        # m_dist collapse = < 50% of the high-phase median
MARGIN_RESTORE = 0.8     # restored = >= 80% of high median
GEOM_TOL = 0.2           # "near-constant" geometry = within 20%
COS_SAME = 0.95          # U3 same-representation
COS_NEW = 0.8            # U3 replacement


def hysteresis_transitions(vals) -> list[int]:
    """Episodes at which the registered hysteresis flips state."""
    out, state = [], None
    for i, v in enumerate(vals):
        if v >= HI:
            if state == "lo":
                out.append(i)
            state = "hi"
        elif v <= LO:
            if state == "hi":
                out.append(i)
            state = "lo"
    return out


def segments(vals) -> list[tuple[str, int, int]]:
    """(phase, first_index, last_index) maximal runs of hi/lo (excluding middle)."""
    segs, state, start = [], None, 0
    for i, v in enumerate(vals):
        s = "hi" if v >= HI else ("lo" if v <= LO else "mid")
        if s != state:
            if state in ("hi", "lo"):
                segs.append((state, start, i - 1))
            if s in ("hi", "lo"):
                state, start = s, i
            else:
                state = s if s == "mid" else state
                if state == "mid":
                    state, start = None, i + 1
    if state in ("hi", "lo"):
        segs.append((state, start, len(vals) - 1))
    return segs


def cos(a, b) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def analyse_run(d: Path) -> dict:
    tag = d.name
    s = json.loads((d / "arm_summary.json").read_text())
    ep = pd.read_csv(d / "per_episode.csv")
    ep = ep[ep.frozen.astype(bool) != True].sort_values("episode")  # noqa: E712
    delayed = ep.delayed_intact.values
    episodes = ep.episode.values
    trans = hysteresis_transitions(delayed)
    segs = segments(delayed)
    hi_segs = [sg for sg in segs if sg[0] == "hi"]
    lo_segs = [sg for sg in segs if sg[0] == "lo"]

    probes = pd.read_csv(d / "dense_probes.csv").sort_values("episode")
    dirs = None
    if (d / "dense_dirs.npz").exists():
        z = np.load(d / "dense_dirs.npz")
        dirs = {"episode": z["episode"], "wp": z["wp"], "dmu": z["dmu"]}

    # ---- phenotype classification (registered: the three are not interchangeable)
    subst = probes.subst_diff_follows.values
    dec = probes.decode_pre_response.values
    t_subst = hysteresis_transitions(subst)
    t_dec = hysteresis_transitions(dec)
    phen = {
        "behavioural_transitions": len(trans),
        "functional_transitions": len(t_subst),
        "representational_transitions": len(t_dec),
        "decode_range": [float(np.nanmin(dec)), float(np.nanmax(dec))],
        "subst_range": [float(np.nanmin(subst)), float(np.nanmax(subst))],
    }

    # ---- U3: cycle-comparative geometry across successive high phases
    u3 = {"n_high_phases": len(hi_segs)}
    if dirs is not None and len(hi_segs) >= 2:
        wp_last, dmu_last, bvm_last = [], [], []
        for sg in hi_segs:
            eps_in = episodes[sg[1]:sg[2] + 1]
            m = [i for i, e in enumerate(dirs["episode"]) if e in set(eps_in)]
            if m:
                k = m[-1]
                wp_last.append(dirs["wp"][k])
                dmu_last.append(dirs["dmu"][k])
                pv = probes[probes.episode == dirs["episode"][k]]
                bvm_last.append(float(pv.boundary_vs_mid.iloc[0]) if len(pv) else np.nan)
        if len(wp_last) >= 2:
            c_wp = [cos(wp_last[i], wp_last[i + 1])
                    for i in range(len(wp_last) - 1)]
            c_dmu = [cos(dmu_last[i], dmu_last[i + 1])
                     for i in range(len(dmu_last) - 1)]
            ndmu = [float(np.linalg.norm(dmu_last[i] - dmu_last[i + 1]))
                    for i in range(len(dmu_last) - 1)]
            u3.update({
                "cos_wp_successive_highs": c_wp,
                "cos_dmu_successive_highs": c_dmu,
                "dmu_shift_norm": ndmu,
                "boundary_vs_mid_lasts": bvm_last,
                "min_cos_wp": float(np.nanmin(c_wp)),
            })

    # ---- instrumental ordering at each high->low crossing
    orderings = []
    for si, sg in enumerate(lo_segs):
        drop_ep = int(episodes[sg[1]])
        # high-phase baseline: median over the dense probes within hi phases
        hi_eps = set()
        for h in hi_segs:
            hi_eps.update(int(e) for e in episodes[h[1]:h[2] + 1])
        base = probes[probes.episode.isin(hi_eps)]
        if len(base) == 0:
            continue
        base_m = float(base.m_dist.median())
        # probes within the transition window: 2 scheduled probes before drop_ep
        # through the first 2 after
        win = probes[(probes.episode >= drop_ep - 20) & (probes.episode <= drop_ep + 20)]
        if len(win) == 0:
            continue
        first = win.iloc[0]
        crossed = {
            "m_dist": bool(first.m_dist < MARGIN_DROP * base_m),
            "decode": bool(first.decode_pre_response < DECODE_OK),
            "subst": bool(first.subst_diff_follows < 0.9),
            "disp_change": bool(abs(first.disp_state - base.disp_state.median())
                                > GEOM_TOL * base.disp_state.median()),
            "class_sep_change": bool(abs(first.class_sep_state - base.class_sep_state.median())
                                     > GEOM_TOL * base.class_sep_state.median()),
        }
        orderings.append({
            "drop_episode": drop_ep, "probe_episode": int(first.episode),
            "crossed_at_first_post_probe": crossed,
            "m_dist_first": float(first.m_dist), "m_dist_base": base_m,
            "decode_first": float(first.decode_pre_response),
            "subst_first": float(first.subst_diff_follows),
        })

    # low-phase vs high-phase margin distance (M1/M3 evidence)
    m_lo, m_hi = [], []
    for sg in lo_segs:
        lo_eps = set(int(e) for e in episodes[sg[1]:sg[2] + 1])
        pv = probes[probes.episode.isin(lo_eps)]
        m_lo.extend(pv.m_dist.dropna().tolist())
    for sg in hi_segs:
        hi_eps = set(int(e) for e in episodes[sg[1]:sg[2] + 1])
        pv = probes[probes.episode.isin(hi_eps)]
        m_hi.extend(pv.m_dist.dropna().tolist())
    m_lo = float(np.median(m_lo)) if m_lo else float("nan")
    m_hi = float(np.median(m_hi)) if m_hi else float("nan")

    # ---- U2 interrupt report
    u2 = s.get("interrupt") or {}
    u2_rep = {}
    if u2:
        reps = u2.get("repeats", [])
        fh = u2.get("frozen_hold", [])
        u2_rep = {
            "e_star": u2.get("e_star"),
            "battery_delayed_at_low": u2.get("battery_delayed"),
            "repeats_identical": (len({tuple(sorted(r.items())) for r in reps}) == 1
                                  if reps else None),
            "subst_error_bar": (float(max(r.get("subst_diff_follows", 0) for r in reps)
                                      - min(r.get("subst_diff_follows", 0) for r in reps))
                                if reps else None),
            "frozen_hold_access_returned": bool(
                any(r["delayed_intact"] >= HI for r in fh)),
            "frozen_hold_max_delayed": (float(max(r["delayed_intact"] for r in fh))
                                        if fh else None),
            "params_restored_bitexact": u2.get("params_restored_bitexact"),
        }

    # ---- M verdict (first-match order: M1, M2, M3)
    geom_stable = (m_lo == m_lo and m_hi == m_hi and m_lo >= MARGIN_RESTORE * m_hi)
    same_rep = (u3.get("min_cos_wp") is not None and u3["min_cos_wp"] >= COS_SAME)
    new_rep = (u3.get("min_cos_wp") is not None and u3["min_cos_wp"] < COS_NEW)
    m_dist_collapsed = any(o["crossed_at_first_post_probe"]["m_dist"] for o in orderings)
    if len(trans) >= 3 and m_dist_collapsed and geom_restores(probes, ep, segs) and same_rep:
        verdict = "M1_boundary_crossing"
    elif len(trans) >= 3 and new_rep:
        verdict = "M2_representation_replacement"
    elif len(trans) >= 3 and geom_stable:
        verdict = "M3_utilisation_only_instability"
    else:
        verdict = "M4_mixed_or_other"

    out = {
        "tag": tag, "seed": s["seed"], "verdict": verdict,
        "n_transitions": len(trans),
        "phenotype": phen, "orderings": orderings, "u3": u3, "u2": u2_rep,
        "m_dist_median_high": m_hi, "m_dist_median_low": m_lo,
        "q_check": s.get("q_determinism_check"),
    }
    return out


def geom_restores(probes, ep, segs) -> bool:
    """After each low->high regain probe, m_dist is restored (>= 80% of the
    preceding high-phase median)."""
    ok = []
    hi_prev = None
    for sg in segs:
        hi_eps = set(int(e) for e in ep.episode.values[sg[1]:sg[2] + 1])
        pv = probes[probes.episode.isin(hi_eps)]
        med = float(pv.m_dist.median()) if len(pv) else np.nan
        if sg[0] == "hi":
            hi_prev = med
        elif sg[0] == "lo" and hi_prev == hi_prev:
            ok.append(bool(pv.m_dist.max() >= MARGIN_RESTORE * hi_prev))
    return all(ok) if ok else False


def main() -> int:
    runs = []
    for d in sorted(BASE.glob("*__*/arm_summary.json")):
        runs.append(analyse_run(d.parent))
    if not runs:
        print("no runs found")
        return 1
    t = pd.DataFrame([{k: v for k, v in r.items()
                       if k not in ("phenotype", "orderings", "u3", "u2")}
                      for r in runs])
    pd.set_option("display.width", 220)
    print(t.to_string(index=False))
    print("\n--- phenotypes ---")
    for r in runs:
        p = r["phenotype"]
        print(f"seed {r['seed']:>2}: behav={p['behavioural_transitions']} "
              f"func={p['functional_transitions']} repr={p['representational_transitions']} "
              f"decode_range={p['decode_range']} subst_range={p['subst_range']}")
    print("\n--- U2 interrupts ---")
    for r in runs:
        print(f"seed {r['seed']:>2}: {r['u2']}")
    print("\n--- U3 ---")
    for r in runs:
        u3 = r["u3"]
        print(f"seed {r['seed']:>2}: min_cos_wp={u3.get('min_cos_wp')} "
              f"dmu_shift={u3.get('dmu_shift_norm')}")
    print("\n--- orderings (first high->low) ---")
    for r in runs:
        if r["orderings"]:
            o = r["orderings"][0]
            print(f"seed {r['seed']:>2}: drop@{o['drop_episode']} "
                  f"crossed={o['crossed_at_first_post_probe']} "
                  f"m_dist {o['m_dist_first']:.4f} (base {o['m_dist_base']:.4f})")
    print("\n--- verdict tally ---")
    from collections import Counter
    print(Counter(r["verdict"] for r in runs))

    import json as _j
    (BASE / "analysis.json").write_text(_j.dumps(runs, indent=2, default=float))
    print(f"wrote {BASE / 'analysis.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
