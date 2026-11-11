"""M1-a scorer (TECHNICAL_REPORT §10): registered E1 / E1b verdicts.

E1 (primary): per seed, P = {immediate_test_accuracy, delayed_test_accuracy,
delayed_intact, delayed_state_reset} of the resume arm must equal the
control's exactly. Arm PASS: 4/4; 3/4 -> PASS-with-drift-adjudication
(max|dP| reported); <= 2/4 -> FAIL.

E1b (secondary, seed 0): the mid-episode (episode-restart) resume arm must
equal the E1 control's seed-0 P exactly; its verdict is independent of E1.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BASE = Path("results/m1_resume")
KEYS = ("immediate_test_accuracy", "delayed_test_accuracy",
        "delayed_intact", "delayed_state_reset")
SEEDS = (0, 1, 2, 3)


def load(dir_name: str) -> dict:
    path = BASE / dir_name / "probe_summary.json"
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)
    print("== E1: episode-boundary resume exactness (registered) ==")
    passes, rows = 0, []
    for seed in SEEDS:
        control = load(f"E1_seed{seed:03d}__control")
        resumed = load(f"E1_seed{seed:03d}__resume")
        diffs = {k: float(resumed[k]) - float(control[k]) for k in KEYS}
        ok = all(resumed[k] == control[k] for k in KEYS)
        passes += int(ok)
        rows.append({"seed": seed, "pass": ok,
                     **{f"d_{k}": v for k, v in diffs.items()}})
        print(f"seed {seed}: {'PASS' if ok else 'FAIL'}  "
              + "  ".join(f"{k}={control[k]:.6f}/{resumed[k]:.6f}" for k in KEYS))

    if passes == 4:
        e1 = "PASS"
    elif passes == 3:
        max_d = max(abs(r[f"d_{k}"]) for r in rows for k in KEYS)
        e1 = f"PASS-with-drift-adjudication (3/4; max|dP|={max_d:.6g})"
    else:
        e1 = f"FAIL ({passes}/4)"
    print(f"\nV-E1: {passes}/4 seeds exact -> {e1}")

    print("\n== E1b: mid-episode resume (episode restart), seed 0 ==")
    try:
        control0 = load("E1_seed000__control")
        midep = load("E1b_seed000__resume_midep")
        ok = all(midep[k] == control0[k] for k in KEYS)
        diffs = {k: float(midep[k]) - float(control0[k]) for k in KEYS}
        print("E1b: " + ("PASS" if ok else "FAIL")
              + ("" if ok else f"  diffs={diffs}"))
    except FileNotFoundError as exc:
        print(f"E1b: not scored ({exc})")


if __name__ == "__main__":
    main()
