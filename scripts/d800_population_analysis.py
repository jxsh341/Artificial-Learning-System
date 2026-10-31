#!/usr/bin/env python
"""D800 population-reliability analysis (pre-registered, report §9).

Reads per-seed summaries from the 20-seed population run
(`results/d800_population/curriculum/D800_interleaved/seed_*`) and the
original 3-seed run (`results/d800_interleaved/...`), keeping the two samples
VISIBLE SEPARATELY (no silent pooling), and reports:

- observed success proportion (delayed_test_accuracy >= 0.9) per sample and
  combined, with exact Clopper-Pearson two-sided 95% intervals (bisection on
  the binomial CDF — no external dependencies);
- immediate accuracy, state-reset, and unseen-size values per seed.

Pre-registered success criterion: the ORIGINAL one — delayed accuracy >= 0.9
per seed; observed proportion reported with CI. Immediate accuracy is
reported separately (not part of the success criterion).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

POP = ROOT / "results" / "d800_population" / "curriculum" / "D800_interleaved"
ORIG = ROOT / "results" / "d800_interleaved" / "curriculum" / "D800_interleaved"
THRESHOLD = 0.9


def binom_cdf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 1.0 if k < 0 else 0.0 if k >= 0 else 1.0
    total = 0.0
    coef = 1.0
    for i in range(min(k, n) + 1):
        if i > 0:
            coef = coef * (n - i + 1) / i
        total += coef * (p ** i) * ((1 - p) ** (n - i))
    return min(max(total, 0.0), 1.0)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided CI. lower: F(k-1, p) = 1 - alpha/2 (F decreasing in p);
    upper: F(k, p) = alpha/2. F is the binomial CDF."""
    if n == 0:
        return (0.0, 1.0)
    lo, hi = 0.0, 1.0
    if k > 0:
        a, b = 0.0, 1.0
        for _ in range(200):
            mid = (a + b) / 2
            if binom_cdf(k - 1, n, mid) > 1 - alpha / 2:
                a = mid  # F decreasing in p: crossing lies above mid
            else:
                b = mid
        lo = (a + b) / 2
    if k < n:
        a, b = 0.0, 1.0
        for _ in range(200):
            mid = (a + b) / 2
            if binom_cdf(k, n, mid) > alpha / 2:
                a = mid
            else:
                b = mid
        hi = (a + b) / 2
    return (lo, hi)


def load_seeds(base: Path) -> list[dict]:
    rows = []
    for sd in sorted(base.glob("seed_*")):
        p = sd / "seed_summary.json"
        if p.exists():
            rows.append(json.loads(p.read_text()))
    return rows


def fmt_interval(k: int, n: int) -> str:
    lo, hi = clopper_pearson(k, n)
    return f"{k}/{n} = {k / n:.3f} (95% CI {lo:.3f}-{hi:.3f})"


def main() -> int:
    pop = load_seeds(POP)
    orig = load_seeds(ORIG)
    if not pop:
        print("No population seed summaries found yet.")
        return 1

    for label, seeds in (("original (kept visible)", orig), ("population (new seeds)", pop)):
        print(f"\n=== {label}: {len(seeds)} seeds ===")
        for s in seeds:
            u = s.get("graph_generalization_unseen_size")
            print(f"  seed {s['seed']:>2}: delayed={s['delayed_test_accuracy']:.3f} "
                  f"immediate={s['immediate_test_accuracy']:.3f} "
                  f"reset={s.get('delayed_state_reset', float('nan')):.3f} "
                  f"unseen_size={u if u is not None else float('nan'):.3f}")
        k = sum(1 for s in seeds if s["delayed_test_accuracy"] >= THRESHOLD)
        n = len(seeds)
        print(f"  success (delayed >= {THRESHOLD}): {fmt_interval(k, n)}")
        imm = [s["immediate_test_accuracy"] for s in seeds]
        print(f"  immediate accuracy: mean {sum(imm) / n:.3f}, min {min(imm):.3f}")
        resets = [s.get("delayed_state_reset", float("nan")) for s in seeds if s["delayed_test_accuracy"] >= THRESHOLD]
        if resets:
            print(f"  state-reset among successful: mean {sum(resets) / len(resets):.3f} (chance = ~0.5)")

    combined = orig + pop
    kc = sum(1 for s in combined if s["delayed_test_accuracy"] >= THRESHOLD)
    nc = len(combined)
    print(f"\n=== combined (reported separately above; pooled here for power) ===")
    print(f"  success: {fmt_interval(kc, nc)}")

    out = {
        "threshold": THRESHOLD,
        "original": {"n": len(orig), "k": sum(1 for s in orig if s["delayed_test_accuracy"] >= THRESHOLD)},
        "population": {"n": len(pop), "k": sum(1 for s in pop if s["delayed_test_accuracy"] >= THRESHOLD)},
        "combined": {"n": nc, "k": kc,
                     "ci95": clopper_pearson(kc, nc)},
    }
    (ROOT / "results" / "d800_population" / "reliability_summary.json").write_text(
        json.dumps(out, indent=2)
    )
    print("\nwrote results/d800_population/reliability_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
