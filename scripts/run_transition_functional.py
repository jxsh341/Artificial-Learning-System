#!/usr/bin/env python
"""Transition-point causal confirmation (report §8p, evaluation only).

§8o located the behavioural (22-50), functional (<=30 for 12/23) and
informational (50/65) boundaries, but causal utilisation has only been
measured at delays 3/10/30/100 (§8k).  This run takes the full causal
substitution battery to the newly located transition points:

    delays [3, 40, 50, 65]

Delay 3 is an in-run anchor that must reproduce §8k exactly; 40/50/65 are new
causal measurements.  Per checkpoint x delay: behavioural accuracy + reset
control, held-out decode at the response step, and the full substitution
battery (own / same / different-follows / zero / shuffled).

Groups (fixed from §8o's published table): behavioural early-failing
(D_behav <= 30) vs late-failing (>= 40); informational D_info = 50 vs 65.

  python scripts/run_transition_functional.py [--shard 0 --num-shards 6] [--merge]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artificial_system.environment.graph_environment import (  # noqa: E402
    make_generic_graph,
)
from artificial_system.evaluation import graph_classification as ge  # noqa: E402
from artificial_system.evaluation import state_content as sc  # noqa: E402
from artificial_system.experiments import coupling_rescue as cr  # noqa: E402
from artificial_system.models.gnn_ssm import build_model  # noqa: E402
from artificial_system.utils.checkpoints import load_checkpoint  # noqa: E402

DIRS = {
    "original": ROOT / "results" / "d800_interleaved" / "curriculum" / "D800_interleaved",
    "population": ROOT / "results" / "d800_population" / "curriculum" / "D800_interleaved",
}
DELAYS = [3, 40, 50, 65]
O8O_SUMMARY = ROOT / "results" / "delay_transition" / "summary.json"


def checkpoints() -> list[tuple[str, int, Path]]:
    out = []
    for sample, base in DIRS.items():
        for seed_dir in sorted(base.glob("seed_*")):
            rec = seed_dir / "seed_summary.json"
            ck = seed_dir / "checkpoint.pt"
            if not (rec.exists() and ck.exists()):
                continue
            r = json.loads(rec.read_text())
            if r.get("delayed_test_accuracy", 0.0) < 0.9:
                continue
            out.append((sample, int(r["seed"]), ck))
    return out


def load_groups() -> dict[int, dict]:
    """Published §8o per-checkpoint boundaries."""
    if not O8O_SUMMARY.exists():
        return {}
    s = json.loads(O8O_SUMMARY.read_text())
    groups = {}
    for key, v in s.get("per_checkpoint", {}).items():
        seed = int(key.split("seed")[1])
        groups[seed] = {"D_behav": v.get("D_behavioural"), "D_info": v.get("D_info")}
    return groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/transition_functional")
    ap.add_argument("--delays", type=int, nargs="*", default=DELAYS)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        parts = sorted(out_dir.glob("transition_functional_shard*.csv"))
        df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        df = df.sort_values(["sample", "seed", "delay"]).reset_index(drop=True)
        df.to_csv(out_dir / "transition_functional.csv", index=False)
        _summarise(df, out_dir)
        return 0

    csv_path = out_dir / f"transition_functional_shard{args.shard}.csv"
    ckpts = checkpoints()
    mine = [c for i, c in enumerate(ckpts) if i % args.num_shards == args.shard]
    print(f"[shard {args.shard}/{args.num_shards}] {len(mine)} of {len(ckpts)} "
          f"checkpoints x delays {args.delays}", flush=True)

    rows: list[dict] = []
    for sample, seed, ck_path in mine:
        ck = load_checkpoint(ck_path)
        cfg = ck["config"]
        task_cfg = cfg.get("task", {})
        marker_dim = int(task_cfg.get("marker_dim", 2))
        datasets = cr.build_datasets_from_cfg(cfg)
        model = build_model(
            cfg, node_feature_dim=int(cfg.get("graph", {}).get("node_feature_dim", 2)),
            edge_feature_dim=0, num_actions=2, num_nodes=None,
        )
        model.load_state_dict(ck["model_state"])
        model.eval()
        blank = None
        if task_cfg.get("blank_mode", "no_markers") == "generic":
            blank = make_generic_graph(
                int(task_cfg.get("num_nodes", 100)), marker_dim, seed * 2000 + 2
            )

        for delay in args.delays:
            beh = ge.probe_delayed_intact_vs_reset(
                model, datasets["test"], delay=delay, batch_size=32,
                marker_dim=marker_dim, generic_blank=blank,
            )
            states = sc.capture_trial_states(
                model, datasets["test"], delay=delay, batch_size=args.batch_size,
                marker_dim=marker_dim, generic_blank=blank,
            )
            model.eval()
            dec = sc.ridge_decode_cv(
                states["pre_response"].numpy(), states["_label"].numpy(), seed=seed
            )
            _, subst = sc.state_substitution(
                model, datasets["test"], delay=delay, batch_size=args.batch_size,
                marker_dim=marker_dim, generic_blank=blank, seed=seed,
            )
            row = {
                "sample": sample, "seed": seed, "delay": int(delay),
                "delayed_intact": float(beh["delayed_intact"]),
                "delayed_reset": float(beh["delayed_state_reset"]),
                "decode_pre_response": float(dec["accuracy"]),
                "decode_baseline": float(dec["majority_baseline"]),
                "mean_state_norm": float(states["pre_response"].norm(dim=-1).mean()),
                "across_instance_std": float(states["pre_response"].numpy().std(axis=0).mean()),
                "subst_same_follows": float(subst["subst_same_follows"]),
                "subst_diff_follows": float(subst["subst_diff_follows"]),
                "subst_zero_accuracy": float(subst["subst_zero_accuracy"]),
                "subst_shuffled_accuracy": float(subst["subst_shuffled_accuracy"]),
            }
            rows.append(row)
            print(f"  {sample:>10} seed {seed:>2} delay {delay:>2}: "
                  f"behav={row['delayed_intact']:.3f} decode={row['decode_pre_response']:.3f} "
                  f"same={row['subst_same_follows']:.3f} follows={row['subst_diff_follows']:.3f} "
                  f"zero={row['subst_zero_accuracy']:.3f} shuf={row['subst_shuffled_accuracy']:.3f} "
                  f"disp={row['across_instance_std']:.4f}", flush=True)
            pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"wrote {csv_path}", flush=True)
    return 0


def _functional(row) -> bool:
    return bool(row.subst_diff_follows >= 0.9 and row.subst_same_follows >= 0.9
                and row.subst_zero_accuracy <= 0.65 and row.subst_shuffled_accuracy <= 0.65)


def _summarise(df: pd.DataFrame, out_dir: Path) -> None:
    groups = load_groups()
    df = df.copy()
    df["functional"] = df.apply(_functional, axis=1)
    df["informational"] = df.decode_pre_response >= 0.95
    df["behavioural"] = df.delayed_intact >= 0.9
    df["D_behav"] = df.seed.map(lambda s: (groups.get(s) or {}).get("D_behav"))
    df["D_info"] = df.seed.map(lambda s: (groups.get(s) or {}).get("D_info"))
    df.to_csv(out_dir / "transition_functional.csv", index=False)

    anchor = df[df.delay == 3]
    anchor_check = {
        "n": int(len(anchor)),
        "behavioural_all": bool((anchor.behavioural).all()),
        "informational_all": bool((anchor.informational).all()),
        "functional_all": bool((anchor.functional).all()),
    }

    per_ckpt: dict = {}
    for (sample, seed), g in df.groupby(["sample", "seed"]):
        per_ckpt[f"{sample}_seed{seed:03d}"] = {
            "D_behav": int(g.D_behav.iloc[0]) if g.D_behav.notna().any() else None,
            "D_info": int(g.D_info.iloc[0]) if g.D_info.notna().any() else None,
            "per_delay": {
                int(r.delay): {
                    "behavioural": float(r.delayed_intact),
                    "informational_decode": float(r.decode_pre_response),
                    "functional_follows": float(r.subst_diff_follows),
                    "functional_same": float(r.subst_same_follows),
                    "zero": float(r.subst_zero_accuracy),
                    "shuffled": float(r.subst_shuffled_accuracy),
                    "functional_pass": bool(r.functional),
                    "dispersion": float(r.across_instance_std),
                } for r in g.itertuples()
            },
        }

    def agg(g: pd.DataFrame) -> dict:
        return {
            "n": int(len(g)),
            "functional": int(g.functional.sum()),
            "informational": int(g.informational.sum()),
            "behavioural": int(g.behavioural.sum()),
            "follows_mean": float(g.subst_diff_follows.mean()),
            "follows_min": float(g.subst_diff_follows.min()),
            "same_mean": float(g.subst_same_follows.mean()),
            "decode_mean": float(g.decode_pre_response.mean()),
            "zero_mean": float(g.subst_zero_accuracy.mean()),
            "shuffled_mean": float(g.subst_shuffled_accuracy.mean()),
            "dispersion_mean": float(g.across_instance_std.mean()),
        }

    by_delay = {int(d): agg(g) for d, g in df.groupby("delay")}
    # primary comparisons (pre-registered)
    d40 = df[df.delay == 40]
    early = d40[d40.D_behav <= 30]
    late = d40[d40.D_behav >= 40]
    comparison_40 = {
        "early_failing_group": agg(early),
        "late_failing_group": agg(late),
        "rule_R1": bool((~early.functional).all()) if len(early) else None,
        "rule_R2_checkpoints": [f"{r.sample}_seed{int(r.seed):03d}"
                                for r in early[early.functional].itertuples()],
    }
    comparison_50 = {
        "D_info50_group": agg(df[(df.delay == 50) & (df.D_info == 50)]),
        "D_info65_group": agg(df[(df.delay == 50) & (df.D_info == 65)]),
    }
    comparison_65 = {
        "D_info50_group": agg(df[(df.delay == 65) & (df.D_info == 50)]),
        "D_info65_group": agg(df[(df.delay == 65) & (df.D_info == 65)]),
    }

    summary = {
        "delays": [int(d) for d in args.delays],
        "n_checkpoints": int(df.seed.nunique()),
        "anchor_delay3": anchor_check,
        "by_delay": by_delay,
        "comparison_40": comparison_40,
        "comparison_50": comparison_50,
        "comparison_65": comparison_65,
        "per_checkpoint": per_ckpt,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    print(f"\nanchor (delay 3): {anchor_check}")
    print("\nby delay:")
    for d, a in by_delay.items():
        print(f"  delay {d:>2}: functional {a['functional']}/{a['n']} "
              f"informational {a['informational']}/{a['n']} "
              f"behavioural {a['behavioural']}/{a['n']} | follows mean={a['follows_mean']:.3f} "
              f"same={a['same_mean']:.3f} zero={a['zero_mean']:.3f} shuf={a['shuffled_mean']:.3f}")
    print("\nprimary comparison at 40 (early-failing vs late-failing):")
    for k in ("early_failing_group", "late_failing_group"):
        a = comparison_40[k]
        print(f"  {k}: functional {a['functional']}/{a['n']} "
              f"follows={a['follows_mean']:.3f} (min {a['follows_min']:.3f}) "
              f"decode={a['decode_mean']:.3f} disp={a['dispersion_mean']:.4f}")
    print(f"  R1 (all early-failing fail functionally at 40): {comparison_40['rule_R1']}")
    print(f"  R2 (early-failing but functional at 40): {comparison_40['rule_R2_checkpoints']}")
    print("\nprimary comparison at 50 / 65 by D_info group:")
    for label, comp in (("50", comparison_50), ("65", comparison_65)):
        for k, a in comp.items():
            print(f"  delay {label} {k}: functional {a['functional']}/{a['n']} "
                  f"follows={a['follows_mean']:.3f} decode={a['decode_mean']:.3f} "
                  f"disp={a['dispersion_mean']:.4f}")
    print(f"\nwrote {out_dir / 'transition_functional.csv'} and {out_dir / 'summary.json'}")


if __name__ == "__main__":
    sys.exit(main())
