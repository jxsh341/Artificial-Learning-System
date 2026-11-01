#!/usr/bin/env python
"""Gradient isolation at delay 30 (report §8n).

§8m showed the delay-30 rescue collapses under encoder-only joint updates even
when the two objectives are balanced in response events (0/12, within 10
pairs).  This run isolates WHICH objective is destabilising, using the same 12
captured checkpoints and the same per-side doses as §8m:

- ``delayed_only``   : 100 delayed episodes at delay 30, nothing else;
- ``immediate_only`` : 100 immediate steps (sequence_length=1, delay 0).

Readout frozen (encoder_only) in both.  Together with the captured state
(no updates) and §8m (both), this is the 2x2 decomposition of the joint stage.

At the first probe with delayed accuracy < 0.9 an explicit decodability
measurement is taken, to distinguish boundary translation (decode intact) from
representational loss (decode collapsed).

  python scripts/run_isolation_arms.py --arms delayed_only,immediate_only \
      [--shard 0 --num-shards 6] [--merge]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artificial_system.environment.graph_environment import (  # noqa: E402
    GraphClassificationEnvironment,
    make_generic_graph,
)
from artificial_system.evaluation import state_content as sc  # noqa: E402
from artificial_system.experiments import coupling_rescue as cr  # noqa: E402
from artificial_system.models.gnn_ssm import build_model  # noqa: E402
from artificial_system.training.trainer import Trainer  # noqa: E402

STAGE_B_DIR = ROOT / "results" / "delay30_h7b"
ARMS = ("delayed_only", "immediate_only")
PROBE_AT = sorted({1, 2, 3, 4, 5, *range(10, 101, 10)})


def captured_checkpoints() -> list[tuple[str, Path]]:
    out = []
    for d in sorted(STAGE_B_DIR.glob("*__h7b_capture_alternate__t30")):
        p = d / "captured_checkpoint.pt"
        if p.exists():
            out.append((d.name.removesuffix("__h7b_capture_alternate__t30"), p))
    return out


def origin_checkpoint(seed: int) -> str:
    for base in ("d800_population", "d800_interleaved"):
        p = ROOT / "results" / base / "curriculum" / "D800_interleaved" / f"seed_{seed:03d}" / "checkpoint.pt"
        if p.exists():
            return str(p.relative_to(ROOT))
    return ""


@torch.no_grad()
def quick_decode(model, test_ds, delay, marker_dim, generic_blank, seed) -> dict:
    """Cheap representational check at the response step (no substitution)."""
    model.eval()
    states = sc.capture_trial_states(
        model, test_ds, delay=delay, batch_size=50,
        marker_dim=marker_dim, generic_blank=generic_blank,
    )
    model.eval()
    X = states["pre_response"].numpy()
    labels = states["_label"].numpy()
    dec = sc.ridge_decode_cv(X, labels, lam=1.0, seed=seed)
    margins = states["_signed_margin"].numpy()
    signed = np.where(labels >= 1, margins, -margins)
    return {
        "decode_pre_response": float(dec["accuracy"]),
        "decode_baseline": float(dec["majority_baseline"]),
        "state_norm": float(np.linalg.norm(X, axis=1).mean()),
        "signed_margin_mean": float(signed.mean()),
    }


def run_one(ck_path: Path, arm: str, run_dir: Path, episodes: int) -> dict:
    assert arm in ARMS
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    seed = int(ck["seed"])
    delay = int(ck.get("delay", 30))
    capture = ck.get("capture")

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    task_cfg = dict(cfg.get("task", {}))
    train_cfg = dict(cfg.get("training", {}))
    task_cfg["delay"] = delay
    marker_dim = int(task_cfg.get("marker_dim", 2))

    datasets = cr.build_datasets_from_cfg(cfg)
    test_ds = datasets["test"]
    model = build_model(
        cfg, node_feature_dim=int(cfg.get("graph", {}).get("node_feature_dim", 2)),
        edge_feature_dim=0, num_actions=2, num_nodes=None,
    )
    model.load_state_dict(ck["model_state"])

    env_del = GraphClassificationEnvironment(datasets["train"], task_cfg, seed=seed * 2000 + 2)
    imm_cfg = dict(task_cfg)
    imm_cfg["delay"] = 0
    env_imm = GraphClassificationEnvironment(datasets["train"], imm_cfg, seed=seed * 2000 + 1)
    rng_states = ck.get("rng_states") or {}
    if "env_del" in rng_states:
        env_del.rng.set_state(rng_states["env_del"])
    if "env_imm" in rng_states:
        env_imm.rng.set_state(rng_states["env_imm"])

    generic_blank = None
    if task_cfg.get("blank_mode", "no_markers") == "generic":
        generic_blank = make_generic_graph(
            int(task_cfg.get("num_nodes", 100)),
            int(cfg.get("graph", {}).get("node_feature_dim", 2)),
            seed * 2000 + 2,
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    cr.set_trainables(model, "encoder_only")
    tr_del = Trainer(model, env_del, optimizer, train_cfg)
    tr_step = Trainer(model, env_imm, optimizer, {**train_cfg, "sequence_length": 1})

    pre_summary = cr._state_summary(
        model, test_ds, delay, marker_dim, generic_blank, seed, run_dir, "pre"
    )
    snapshot_start = cr.param_group_snapshot(model)

    rows: list[dict] = []

    def probe(tag: str, done: int) -> None:
        quick = cr.quick_probe(model, test_ds, delay, marker_dim, generic_blank)
        rows.append({"phase": tag, "episodes_done": done, **quick})

    probe("pre", 0)

    collapse_info: dict | None = None
    done = 0
    for ep in range(1, episodes + 1):
        if arm == "delayed_only":
            tr_del.train(1)
        else:
            tr_step.train(1)
        done = ep
        if ep in PROBE_AT:
            probe(f"ep{ep}", done)
            last = rows[-1]
            if collapse_info is None and last["delayed_intact"] < 0.9:
                # earliest behavioural collapse: measure the representation now
                dec = quick_decode(model, test_ds, delay, marker_dim, generic_blank, seed)
                collapse_info = {
                    "episode": int(ep),
                    "delayed_at_collapse": float(last["delayed_intact"]),
                    "decode_at_collapse": dec["decode_pre_response"],
                    "state_norm_at_collapse": dec["state_norm"],
                    "signed_margin_at_collapse": dec["signed_margin_mean"],
                }
                rows[-1].update({f"collapse_{k}": v for k, v in dec.items()})
            pd.DataFrame(rows).to_csv(run_dir / "training_probes.csv", index=False)

    post_summary = cr._state_summary(
        model, test_ds, delay, marker_dim, generic_blank, seed, run_dir, "post"
    )
    stage_summary = post_summary  # single-stage run
    battery = cr._final_battery(model, test_ds, delay, marker_dim, generic_blank, seed)
    snapshot_end = cr.param_group_snapshot(model)
    rubric = cr.success_rubric(pre_summary, post_summary, battery)

    from artificial_system.evaluation import readout_geometry as rg
    post_geometry = rg.analyse(model, test_ds, delay, marker_dim, generic_blank, seed)

    summary = {
        "arm": arm,
        "seed": seed,
        "delay": int(delay),
        "origin_checkpoint": origin_checkpoint(seed),
        "source_checkpoint": str(ck_path.relative_to(ROOT)),
        "captured_from": str(ck_path.parent.name),
        "recorded_delayed_pre": float(capture["delayed_intact"]) if capture else None,
        "pre_analysis": pre_summary,
        "stage3_analysis": stage_summary,
        "post_analysis": post_summary,
        "battery": battery,
        "capture_info": dict(capture, source_run=ck_path.parent.name) if capture else None,
        "collapse_info": collapse_info,
        "episodes_run": int(episodes),
        "exposure": (
            {"delayed_episodes": int(episodes), "immediate_steps": 0}
            if arm == "delayed_only"
            else {"delayed_episodes": 0, "immediate_steps": int(episodes)}
        ),
        "param_deltas_after_stage3": {k: 0.0 for k in snapshot_start},
        "param_deltas_stage4": cr.param_group_deltas(snapshot_start, snapshot_end),
        "rubric": rubric,
        "post_geometry": post_geometry,
        "episodes_total": int(episodes),
    }
    pd.DataFrame(rows).to_csv(run_dir / "training_probes.csv", index=False)
    with (run_dir / "arm_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary


def _merge(out_dir: Path) -> None:
    merged: dict = {}
    for f in sorted(out_dir.glob("*__*/arm_summary.json")):
        merged[f.parent.name] = json.loads(f.read_text())
    (out_dir / "summary.json").write_text(json.dumps(merged, indent=2, default=float))
    print(f"merged {len(merged)} runs -> {out_dir / 'summary.json'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/delay30_isolation")
    ap.add_argument("--arms", default="delayed_only,immediate_only")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge:
        _merge(out_dir)
        return 0

    ckpts = captured_checkpoints()
    jobs = [(tag, arm) for i, (tag, ck) in enumerate(ckpts)
            for arm in args.arms.split(",")
            if i % args.num_shards == args.shard]
    print(f"[shard {args.shard}/{args.num_shards}] {len(jobs)} of "
          f"{len(ckpts) * len(args.arms.split(','))} runs, {args.episodes} episodes", flush=True)

    failures = 0
    for tag, arm in jobs:
        ck_path = dict(captured_checkpoints())[tag]
        run_dir = out_dir / f"{tag}__{arm}"
        if (run_dir / "arm_summary.json").exists():
            print(f"[skip] {tag} {arm} (complete)", flush=True)
            continue
        print(f"[run ] {tag} {arm} -> {run_dir}", flush=True)
        try:
            s = run_one(ck_path, arm, run_dir, args.episodes)
            r, b = s["rubric"], s["battery"]
            ci = s.get("collapse_info")
            print(f"      delayed={b['delayed_intact']:.3f} immediate={b['immediate']:.3f} "
                  f"follows={r['subst_diff_follows']:.3f} "
                  f"decode={s['post_analysis'].get('decode_pre_response', float('nan')):.3f} "
                  f"collapse={('ep' + str(ci['episode']) + ' dec' + format(ci['decode_at_collapse'], '.2f')) if ci else 'never'} "
                  f"ALL={'PASS' if r['all_criteria'] else 'fail'}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"      ERROR in {tag} {arm}: {exc}", flush=True)
            raise
    _merge(out_dir)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
