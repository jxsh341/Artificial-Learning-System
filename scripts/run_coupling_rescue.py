#!/usr/bin/env python
"""Run the H5 state-to-readout coupling rescue (report §8b).

  python scripts/run_coupling_rescue.py \
      [--config configs/experiments/curriculum_study.yaml] \
      [--arms control,treatment_a,treatment_b] \
      [--sources D_interleaved:0,C_staged:1] [--episodes 400]

Arms run sequentially on each specified phenotype-(iii) source checkpoint
(base study). Results: results/coupling_rescue/<source>__<arm>/ with
arm_summary.json + training_probes.csv, and an aggregate summary.json.
Resume guard: completed (source, arm) pairs are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml

from artificial_system.experiments import coupling_rescue as cr
from artificial_system.experiments.runner import load_config
from artificial_system.utils.checkpoints import load_checkpoint

STUDY_RESULTS = ROOT / "results" / "curriculum_study"
REFERENCE_RESULTS = ROOT / "results" / "d800_interleaved"


def _source_paths(source_specs: list[str]) -> list[tuple[str, Path]]:
    out = []
    for spec in source_specs:
        cond, seed = spec.split(":")
        # prefer the base study (phenotype iii), fall back to reference dir
        p = STUDY_RESULTS / "curriculum" / cond / f"seed_{int(seed):03d}" / "checkpoint.pt"
        if not p.exists():
            p = REFERENCE_RESULTS / "curriculum" / cond / f"seed_{int(seed):03d}" / "checkpoint.pt"
        if not p.exists():
            raise FileNotFoundError(spec)
        out.append((f"{cond}_seed{int(seed):03d}", p))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/curriculum_study.yaml")
    ap.add_argument("--arms", default="control,treatment_a,treatment_b")
    ap.add_argument("--sources", default="D_interleaved:0,C_staged:1")
    ap.add_argument("--results-dir", default="results/coupling_rescue")
    ap.add_argument("--probes-every", type=int, default=50)
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    out_dir = ROOT / args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    budget = int(cfg.get("study", {}).get("episodes_total", 400))
    half = budget // 2
    schedule_spec = {
        "stage3": {"episodes": half},
        "stage4": {"episodes": budget - half},
        "probe_every": args.probes_every,
    }

    datasets = cr.build_datasets_from_cfg(cfg)
    ckpts = _source_paths(args.sources.split(","))

    summaries_path = out_dir / "summary.json"
    all_summaries: dict = json.loads(summaries_path.read_text()) if summaries_path.exists() else {}

    failures = 0
    for tag, ck_path in ckpts:
        recorded = json.loads((ck_path.parent / "seed_summary.json").read_text())
        for arm in args.arms.split(","):
            run_dir = out_dir / f"{tag}__{arm}"
            guard = run_dir / "arm_summary.json"
            if guard.exists():
                print(f"[skip] {tag} {arm} (complete)")
                continue
            print(f"[run ] {tag} {arm} -> {run_dir}")
            ck = load_checkpoint(ck_path)
            ck["recorded_delayed"] = recorded.get("delayed_test_accuracy")
            ck["source_path"] = str(ck_path.relative_to(ROOT))
            try:
                summary = cr.run_coupling_arm(
                    cfg, datasets, ck, arm, run_dir, seed=int(ck["seed"]),
                    schedule_spec=schedule_spec,
                )
                all_summaries[f"{tag}__{arm}"] = summary
                r = summary["rubric"]
                print(
                    f"      immediate={summary['battery']['immediate']:.3f} "
                    f"delayed={summary['battery']['delayed_intact']:.3f} "
                    f"subst_diff_follows={r['subst_diff_follows']:.3f} "
                    f"ALL={'PASS' if r['all_criteria'] else 'fail'}"
                )
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"      ERROR in {tag} {arm}: {exc}")
                raise
            summaries_path.write_text(json.dumps(all_summaries, indent=2, default=float))

    print(f"\nresults: {out_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
