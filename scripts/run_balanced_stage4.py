#!/usr/bin/env python
"""Event-balanced joint stage at delay 30 (report §8m).

§8l stage B showed readout-only coupling rescues all 12 delay-30 failures
(capture 12/12, full rescue signature 12/12), but the pre-registered full
rubric is 0/12 because the *episode-based* joint stage erases the rescue while
the readout stays exactly frozen.  Two explanations remain:

  S1 schedule imbalance: 1:1 episodes = 32:1 immediate:delayed response events
     at delay 30, so the delayed task supplies too little gradient;
  S2 intrinsic fragility: the long-delay state cannot be maintained by
     encoder-only joint updates even when events are balanced.

This run changes exactly one variable: the joint stage is balanced in
*response events* instead of episodes.  It starts from each captured
checkpoint (the readout-only rescue point of §8l, saved before stage 4), keeps
the readout frozen (``encoder_only``), and pairs ONE delayed episode — 1
supervised response event per batch element at delay 30 — with ONE immediate
step (``sequence_length=1`` at delay 0, 1 response event), so every pair
delivers equal events.  The number of delayed episodes is the same 100 as in
§8l stage B; only the immediate side changes (100 steps instead of 100
32-step episodes).  Delay 30 is preserved throughout.

  python scripts/run_balanced_stage4.py [--shard 0 --num-shards 6] [--merge]

Outputs results/delay30_balanced/<seed>__h7b_balanced_stage4/ with
arm_summary.json + training_probes.csv, and an aggregate summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artificial_system.environment.graph_environment import (  # noqa: E402
    GraphClassificationEnvironment,
    make_generic_graph,
)
from artificial_system.experiments import coupling_rescue as cr  # noqa: E402
from artificial_system.models.gnn_ssm import build_model  # noqa: E402
from artificial_system.training.trainer import Trainer  # noqa: E402

STAGE_B_DIR = ROOT / "results" / "delay30_h7b"


def captured_checkpoints() -> list[tuple[str, Path]]:
    """(tag, path) for every captured (stage-3-rescued) checkpoint of §8l B."""
    out = []
    for d in sorted(STAGE_B_DIR.glob("*__h7b_capture_alternate__t30")):
        p = d / "captured_checkpoint.pt"
        if p.exists():
            out.append((d.name.removesuffix("__h7b_capture_alternate__t30"), p))
    return out


def origin_checkpoint(seed: int) -> str:
    """Path of the original curriculum checkpoint this seed came from (for the
    geometry join and provenance)."""
    for base in ("d800_population", "d800_interleaved"):
        p = ROOT / "results" / base / "curriculum" / "D800_interleaved" / f"seed_{seed:03d}" / "checkpoint.pt"
        if p.exists():
            return str(p.relative_to(ROOT))
    return ""


def run_one(ck_path: Path, run_dir: Path, pairs: int, probe_every: int, lr: float | None) -> dict:
    import numpy as np

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
    if lr is not None:
        train_cfg["learning_rate"] = float(lr)

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
    # continue the same delayed data stream the §8l stage-4 run started with
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
    # joint stage, readout frozen: encoder + pooling + state generation adapt
    cr.set_trainables(model, "encoder_only")
    tr_del = Trainer(model, env_del, optimizer, train_cfg)
    # 1 step = 1 immediate trial (delay 0) = 1 response event per batch element
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

    done = 0
    while done < pairs:
        burst = min(probe_every, pairs - done)
        cr._run_episodes((None, tr_del, tr_step), burst, "alternating_events")
        done += burst
        probe(f"balanced_ep{done}", done)
        pd.DataFrame(rows).to_csv(run_dir / "training_probes.csv", index=False)
    probe("post_stage", done)

    post_summary = cr._state_summary(
        model, test_ds, delay, marker_dim, generic_blank, seed, run_dir, "post"
    )
    stage_summary = post_summary  # single-stage run: "stage 3" == post
    battery = cr._final_battery(model, test_ds, delay, marker_dim, generic_blank, seed)
    snapshot_end = cr.param_group_snapshot(model)
    rubric = cr.success_rubric(pre_summary, post_summary, battery)

    # secondary diagnostic: what does the erosion look like geometrically?
    from artificial_system.evaluation import readout_geometry as rg
    post_geometry = rg.analyse(model, test_ds, delay, marker_dim, generic_blank, seed)

    batch_size = int(train_cfg.get("batch_size", 16))
    summary = {
        "arm": "h7b_balanced_stage4",
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
        "param_deltas_after_stage3": {k: 0.0 for k in snapshot_start},
        "param_deltas_stage4": cr.param_group_deltas(snapshot_start, snapshot_end),
        "rubric": rubric,
        "pairs": int(pairs),
        "delayed_episodes": int(pairs),
        "immediate_steps": int(pairs),
        "events_per_pair": {"delayed": batch_size, "immediate": batch_size},
        "probe_every_pairs": int(probe_every),
        "post_geometry": post_geometry,
        "episodes_total": int(pairs),
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
    ap.add_argument("--results-dir", default="results/delay30_balanced")
    ap.add_argument("--pairs", type=int, default=100,
                    help="balanced pairs (delayed episodes); default = stage-4 "
                         "episodes of §8l stage B / 2")
    ap.add_argument("--probe-every", type=int, default=10)
    ap.add_argument("--learning-rate", type=float, default=None)
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
    mine = [c for i, c in enumerate(ckpts) if i % args.num_shards == args.shard]
    print(f"[shard {args.shard}/{args.num_shards}] {len(mine)} of {len(ckpts)} "
          f"captured checkpoints, {args.pairs} balanced pairs", flush=True)

    failures = 0
    for tag, ck_path in mine:
        run_dir = out_dir / f"{tag}__h7b_balanced_stage4"
        if (run_dir / "arm_summary.json").exists():
            print(f"[skip] {tag} (complete)", flush=True)
            continue
        print(f"[run ] {tag} -> {run_dir}", flush=True)
        try:
            s = run_one(ck_path, run_dir, args.pairs, args.probe_every, args.learning_rate)
            r, b = s["rubric"], s["battery"]
            print(f"      delayed={b['delayed_intact']:.3f} immediate={b['immediate']:.3f} "
                  f"follows={r['subst_diff_follows']:.3f} "
                  f"decode={s['post_analysis'].get('decode_pre_response', float('nan')):.3f} "
                  f"ALL={'PASS' if r['all_criteria'] else 'fail'}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"      ERROR in {tag}: {exc}", flush=True)
            raise
    _merge(out_dir)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
