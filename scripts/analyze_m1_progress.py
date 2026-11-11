"""M1-c scorer (TECHNICAL_REPORT §10): registered PC1 / PC2 verdicts.

PC1 (decodable): CV accuracy - majority baseline >= 0.15 on >= 6/8 seeds ->
progress decoded; 4-5/8 -> tiered partial; <= 3/8 -> ABSENT-with-evidence.
PC2 (causal): treatment change rate > control change rate on >= 6/8 seeds ->
SUPPORTED; otherwise decodable-but-not-causally-used.
Both terminal outcomes are registered successes of the probe.
"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path("results/m1_progress")
SEEDS = tuple(range(8))
PC1_THRESHOLD = 0.15
PC1_MIN_SEEDS = 6
PC2_MIN_SEEDS = 6


def load(seed: int) -> dict:
    with open(BASE / f"P_seed{seed:03d}" / "probe_summary.json",
              encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    rows = []
    for seed in SEEDS:
        try:
            s = load(seed)
        except FileNotFoundError:
            print(f"seed {seed}: missing probe_summary.json")
            return
        d1 = s["pc1"]["delta_over_baseline"]
        t, c = s["pc2"]["treatment_change_rate"], s["pc2"]["control_change_rate"]
        rows.append({"seed": seed, "pc1_delta": d1, "pc1_pass": d1 >= PC1_THRESHOLD,
                     "treat": t, "ctrl": c, "pc2_pass": t > c})
        print(f"seed {seed}: pc1 acc={s['pc1']['accuracy']:.3f} "
              f"base={s['pc1']['majority_baseline']:.3f} delta={d1:+.3f} | "
              f"pc2 treat={t:.2f} ctrl={c:.2f}")

    n1 = sum(r["pc1_pass"] for r in rows)
    if n1 >= PC1_MIN_SEEDS:
        pc1_v = "PROGRESS DECODED"
    elif n1 >= 4:
        pc1_v = f"tiered partial ({n1}/8)"
    else:
        pc1_v = "ABSENT-with-evidence"
    print(f"\nV-PC1: {n1}/8 seeds delta>={PC1_THRESHOLD} -> {pc1_v}")

    n2 = sum(r["pc2_pass"] for r in rows)
    pc2_v = "SUPPORTED" if n2 >= PC2_MIN_SEEDS else "decodable-but-not-causally-used"
    print(f"V-PC2: {n2}/8 seeds treatment>control -> {pc2_v}")

    # sanity line (registered: not a criterion)
    try:
        accs = [r for r in rows]
        sane = [load(s["seed"])["pc1_remaining_sanity"]["accuracy"] for s in accs]
        print(f"sanity (r decode): accuracies={['%.3f' % a for a in sane]}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
