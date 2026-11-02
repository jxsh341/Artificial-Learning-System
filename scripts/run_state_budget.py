#!/usr/bin/env python
"""Long-delay maintenance under state-budget-constrained encoder updates.

Report section 8q.  Same 12 captured checkpoints and the identical delayed
exposure as section 8n ``delayed_only`` (100 delayed episodes at delay 30,
saved RNG stream, readout frozen), but the encoder update is:

- ``budget_eps{E}``   : per-step backtracking line search -- scale the step by
  alpha in {1, 1/2, ..., 1/128, 0} and accept the largest alpha whose
  cumulative displacement of the response-step delayed states on a fixed
  16-instance held-out probe batch stays within eps of the captured model's
  states (displacement = mean per-instance L2 vs captured);
- ``matched_raw_eps{E}``` : paired control -- the raw delayed-gradient update
  scaled to exactly the per-step accepted norm the budget arm accepted at the
  same step (no state awareness).  Run after the paired budget arm in the same
  process and consumes its per-step norms.

The unconstrained reference arm is section 8n ``delayed_only`` (already
measured; not re-run).

  python scripts/run_state_budget.py --budgets 0.2,0.05 \
      [--shard 0 --num-shards 6] [--merge]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
PROBE_N = 16
ALPHA_LADDER = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.0]
PROBE_AT = sorted({1, 2, 3, 4, 5, *range(10, 101, 10)})


def captured_checkpoints() -> list[tuple[str, Path]]:
    out = []
    for d in sorted(STAGE_B_DIR.glob("*__h7b_capture_alternate__t30")):
        p = d / "captured_checkpoint.pt"
        if p.exists():
            out.append((d.name.removesuffix("__h7b_capture_alternate__t30"), p))
    return out


class BudgetSearch:
    """Intercepts optimizer.step to line-search the encoder update.

    After every parameter update proposal (theta_pre + raw_delta), scales the
    step by the largest alpha in ALPHA_LADDER such that the mean per-instance
    L2 displacement of probe-batch response-step delayed states from the
    captured model's states stays within ``eps``.  alpha = 0 (rejection) is
    always admissible.
    """

    def __init__(self, model, optimizer, eps, probe_states_cap, probe_ds,
                 delay, marker_dim, generic_blank, fast: bool = False):
        self.model = model
        self.optimizer = optimizer
        self.eps = float(eps)
        self.s_cap = probe_states_cap.detach()
        self.probe_ds = probe_ds
        self.delay = int(delay)
        self.marker_dim = marker_dim
        self.generic_blank = generic_blank
        self.fast = bool(fast)
        self.trainable = [p for p in model.parameters() if p.requires_grad]
        self.orig_step = optimizer.step
        self.step_idx = 0
        self.disp0 = 0.0  # cumulative displacement of the accepted state (0 at capture)
        self.log: list[dict] = []
        optimizer.step = self.step  # type: ignore[method-assign]

    @torch.no_grad()
    def _displacement(self) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        st = sc.capture_trial_states(
            self.model, self.probe_ds, delay=self.delay, batch_size=PROBE_N,
            marker_dim=self.marker_dim, generic_blank=self.generic_blank,
        )
        if was_training:
            self.model.train()
        return st["pre_response"].detach().float()

    @torch.no_grad()
    def _disp_for(self, alpha: float, theta_pre, deltas) -> tuple[float, float]:
        """Return (mean displacement, alpha) after setting params = pre + alpha*deltas."""
        for p, t, d in zip(self.trainable, theta_pre, deltas):
            p.copy_(t + alpha * d)
        disp = float((self._displacement() - self.s_cap).norm(dim=-1).mean())
        return disp, alpha

    @torch.no_grad()
    def _accept_fast(self, theta_pre, deltas, disp0: float) -> tuple[float, float]:
        """Crossing-estimate scan; returns (alpha, displacement). Supports α=0."""
        import math

        eps = self.eps
        disp1, _ = self._disp_for(1.0, theta_pre, deltas)
        if disp1 <= eps:
            return 1.0, disp1
        alpha_lo = ALPHA_LADDER[-2]  # 1/128 (α=0 is handled separately)
        disp_lo, _ = self._disp_for(alpha_lo, theta_pre, deltas)
        if disp_lo > eps:
            return 0.0, disp_lo
        span = max(disp1 - disp_lo, 1e-12)
        frac = min(max((eps - disp_lo) / span, 0.0), 1.0)
        a_cross = alpha_lo * (1.0 / alpha_lo) ** frac
        k = int(math.ceil(-math.log2(max(a_cross, alpha_lo))))
        k = min(max(k, 1), 7)
        alpha, disp = None, None
        for kk in range(k, 8):  # walk down if the estimate was optimistic
            a = 2.0 ** -kk
            d, _ = self._disp_for(a, theta_pre, deltas)
            if d <= eps:
                alpha, disp = a, d
                break
        if alpha is None:
            for p, t in zip(self.trainable, theta_pre):
                p.copy_(t)
            return 0.0, disp_lo
        # attempt successive rungs upward while they still qualify
        while alpha < 1.0:
            a_up = min(alpha * 2.0, 1.0)
            d_up, _ = self._disp_for(a_up, theta_pre, deltas)
            if d_up <= eps:
                alpha, disp = a_up, d_up
            else:
                for p, t, d in zip(self.trainable, theta_pre, deltas):
                    p.copy_(t + alpha * d)
                return alpha, disp
        return alpha, disp

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D102
        theta_pre = [p.detach().clone() for p in self.trainable]
        out = self.orig_step(closure)
        deltas = [p.detach() - t for p, t in zip(self.trainable, theta_pre)]

        if self.fast:
            alpha, disp = self._accept_fast(theta_pre, deltas, self.disp0)
        else:
            accepted = None
            for alpha in ALPHA_LADDER:
                d, _ = self._disp_for(alpha, theta_pre, deltas)
                if d <= self.eps:
                    accepted = (alpha, d)
                    break
            if accepted is None:  # unreachable: α=0 restores the accepted state
                for p, t in zip(self.trainable, theta_pre):
                    p.copy_(t)
                accepted = (0.0, float("nan"))
            alpha, disp = accepted
            if alpha == 0.0:
                for p, t in zip(self.trainable, theta_pre):
                    p.copy_(t)
        self.disp0 = disp

        raw_norm = float(torch.sqrt(sum(d.norm() ** 2 for d in deltas)))
        acc_norm = alpha * raw_norm
        self.log.append({
            "step": self.step_idx, "alpha": alpha,
            "raw_step_norm": raw_norm, "accepted_norm": acc_norm,
            "displacement": disp, "eps": self.eps,
        })
        self.step_idx += 1
        return out


class MatchedRawControl:
    """Scales the raw encoder update to the paired budget arm's accepted norm.

    Uses the same per-step index as the budget arm; direction is untouched.
    """

    def __init__(self, model, optimizer, accepted_norms):
        self.model = model
        self.optimizer = optimizer
        self.norms = list(accepted_norms)
        self.trainable = [p for p in model.parameters() if p.requires_grad]
        self.orig_step = optimizer.step
        self.step_idx = 0
        self.log: list[dict] = []
        optimizer.step = self.step  # type: ignore[method-assign]

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D102
        theta_pre = [p.detach().clone() for p in self.trainable]
        out = self.orig_step(closure)
        deltas = [p.detach() - t for p, t in zip(self.trainable, theta_pre)]
        raw_norm = float(torch.sqrt(sum(d.norm() ** 2 for d in deltas)))
        target = self.norms[self.step_idx] if self.step_idx < len(self.norms) else 0.0
        scale = (target / raw_norm) if raw_norm > 0 else 0.0
        for p, t, d in zip(self.trainable, theta_pre, deltas):
            p.copy_(t + scale * d)
        self.log.append({
            "step": self.step_idx, "scale": scale,
            "raw_step_norm": raw_norm, "accepted_norm": raw_norm * scale,
            "target_norm": target,
        })
        self.step_idx += 1
        return out


def run_one(ck_path: Path, tag: str, run_dir: Path, budget: float,
            episodes: int, fast: bool = True) -> tuple[dict, list[float]]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    seed = int(ck["seed"])
    delay = int(ck.get("delay", 30))
    capture = ck.get("capture")

    task_cfg = dict(cfg.get("task", {}))
    task_cfg["delay"] = delay
    marker_dim = int(task_cfg.get("marker_dim", 2))
    train_cfg = dict(cfg.get("training", {}))

    datasets = cr.build_datasets_from_cfg(cfg)
    test_ds = datasets["test"]
    from artificial_system.data.graph_tasks import ConnectivityDataset
    probe_ds = ConnectivityDataset(
        graphs=list(test_ds.graphs[:PROBE_N]), labels=test_ds.labels[:PROBE_N].clone(),
    )

    model = build_model(
        cfg, node_feature_dim=int(cfg.get("graph", {}).get("node_feature_dim", 2)),
        edge_feature_dim=0, num_actions=2, num_nodes=None,
    )
    model.load_state_dict(ck["model_state"])
    cr.set_trainables(model, "encoder_only")

    # captured reference states on the probe batch (model untouched so far)
    model.eval()
    st_cap = sc.capture_trial_states(
        model, probe_ds, delay=delay, batch_size=PROBE_N,
        marker_dim=marker_dim, generic_blank=None,
    )
    s_cap = st_cap["pre_response"].detach().float()

    generic_blank = None
    if task_cfg.get("blank_mode", "no_markers") == "generic":
        generic_blank = make_generic_graph(
            int(task_cfg.get("num_nodes", 100)),
            int(cfg.get("graph", {}).get("node_feature_dim", 2)),
            seed * 2000 + 2,
        )
        model.eval()
        st_cap = sc.capture_trial_states(
            model, probe_ds, delay=delay, batch_size=PROBE_N,
            marker_dim=marker_dim, generic_blank=generic_blank,
        )
        s_cap = st_cap["pre_response"].detach().float()

    env = GraphClassificationEnvironment(datasets["train"], task_cfg, seed=seed * 2000 + 2)
    rng_states = ck.get("rng_states") or {}
    if "env_del" in rng_states:
        env.rng.set_state(rng_states["env_del"])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    pre_summary = cr._state_summary(
        model, test_ds, delay, marker_dim, generic_blank, seed, run_dir, "pre"
    )
    snap_start = cr.param_group_snapshot(model)

    searcher = BudgetSearch(model, optimizer, budget, s_cap, probe_ds,
                            delay, marker_dim, generic_blank, fast=fast)
    trainer = Trainer(model, env, optimizer, train_cfg)

    rows: list[dict] = []

    def probe(phase: str, done: int) -> None:
        quick = cr.quick_probe(model, test_ds, delay, marker_dim, generic_blank)
        rows.append({"phase": phase, "episodes_done": done, **quick})

    probe("pre", 0)
    for ep in range(1, episodes + 1):
        trainer.train(1)
        if ep in PROBE_AT:
            probe(f"ep{ep}", ep)
            pd.DataFrame(rows).to_csv(run_dir / "training_probes.csv", index=False)

    post_summary = cr._state_summary(
        model, test_ds, delay, marker_dim, generic_blank, seed, run_dir, "post"
    )
    battery = cr._final_battery(model, test_ds, delay, marker_dim, generic_blank, seed)
    snap_end = cr.param_group_snapshot(model)
    rubric = cr.success_rubric(pre_summary, post_summary, battery)

    search_log = pd.DataFrame(searcher.log)
    search_log.to_csv(run_dir / "step_log.csv", index=False)
    n_rejected = int((search_log["alpha"] == 0.0).sum())

    from artificial_system.evaluation import readout_geometry as rg
    post_geometry = rg.analyse(model, test_ds, delay, marker_dim, generic_blank, seed)

    summary = {
        "arm": f"budget_eps{budget}",
        "budget": float(budget),
        "scan": "fast_crossing" if fast else "full_ladder",
        "seed": seed,
        "delay": int(delay),
        "source_checkpoint": str(ck_path.relative_to(ROOT)),
        "captured_from": ck_path.parent.name,
        "pre_analysis": pre_summary,
        "post_analysis": post_summary,
        "battery": battery,
        "episodes_run": int(episodes),
        "param_deltas": cr.param_group_deltas(snap_start, snap_end),
        "rubric": rubric,
        "steps": int(searcher.step_idx),
        "alpha_rejected_steps": n_rejected,
        "alpha_rejected_frac": float(n_rejected / max(searcher.step_idx, 1)),
        "accepted_norm_mean": float(search_log["accepted_norm"].mean()),
        "displacement_mean": float(search_log["displacement"].mean()),
        "displacement_max": float(search_log["displacement"].max()),
        "post_geometry": post_geometry,
    }
    pd.DataFrame(rows).to_csv(run_dir / "training_probes.csv", index=False)
    with (run_dir / "arm_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary, [float(x) for x in search_log["accepted_norm"]]


def run_control(ck_path: Path, tag: str, run_dir: Path, budget: float,
                norm_path: Path, episodes: int) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    seed = int(ck["seed"])
    delay = int(ck.get("delay", 30))

    task_cfg = dict(cfg.get("task", {}))
    task_cfg["delay"] = delay
    marker_dim = int(task_cfg.get("marker_dim", 2))
    train_cfg = dict(cfg.get("training", {}))

    datasets = cr.build_datasets_from_cfg(cfg)
    test_ds = datasets["test"]

    model = build_model(
        cfg, node_feature_dim=int(cfg.get("graph", {}).get("node_feature_dim", 2)),
        edge_feature_dim=0, num_actions=2, num_nodes=None,
    )
    model.load_state_dict(ck["model_state"])
    cr.set_trainables(model, "encoder_only")

    generic_blank = None
    if task_cfg.get("blank_mode", "no_markers") == "generic":
        generic_blank = make_generic_graph(
            int(task_cfg.get("num_nodes", 100)),
            int(cfg.get("graph", {}).get("node_feature_dim", 2)),
            seed * 2000 + 2,
        )

    env = GraphClassificationEnvironment(datasets["train"], task_cfg, seed=seed * 2000 + 2)
    rng_states = ck.get("rng_states") or {}
    if "env_del" in rng_states:
        env.rng.set_state(rng_states["env_del"])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    pre_summary = cr._state_summary(
        model, test_ds, delay, marker_dim, generic_blank, seed, run_dir, "pre"
    )
    snap_start = cr.param_group_snapshot(model)

    norms = json.loads(norm_path.read_text())
    control = MatchedRawControl(model, optimizer, norms)
    trainer = Trainer(model, env, optimizer, train_cfg)

    rows: list[dict] = []

    def probe(phase: str, done: int) -> None:
        quick = cr.quick_probe(model, test_ds, delay, marker_dim, generic_blank)
        rows.append({"phase": phase, "episodes_done": done, **quick})

    probe("pre", 0)
    for ep in range(1, episodes + 1):
        trainer.train(1)
        if ep in PROBE_AT:
            probe(f"ep{ep}", ep)
            pd.DataFrame(rows).to_csv(run_dir / "training_probes.csv", index=False)

    post_summary = cr._state_summary(
        model, test_ds, delay, marker_dim, generic_blank, seed, run_dir, "post"
    )
    battery = cr._final_battery(model, test_ds, delay, marker_dim, generic_blank, seed)
    snap_end = cr.param_group_snapshot(model)
    rubric = cr.success_rubric(pre_summary, post_summary, battery)

    step_log = pd.DataFrame(control.log)
    step_log.to_csv(run_dir / "step_log.csv", index=False)

    summary = {
        "arm": f"matched_raw_eps{budget}",
        "budget": float(budget),
        "seed": seed,
        "delay": int(delay),
        "source_checkpoint": str(ck_path.relative_to(ROOT)),
        "captured_from": ck_path.parent.name,
        "pre_analysis": pre_summary,
        "post_analysis": post_summary,
        "battery": battery,
        "episodes_run": int(episodes),
        "param_deltas": cr.param_group_deltas(snap_start, snap_end),
        "rubric": rubric,
        "steps": int(control.step_idx),
        "accepted_norm_mean": float(step_log["accepted_norm"].mean()),
        "target_norm_mean": float(step_log["target_norm"].mean()),
    }
    pd.DataFrame(rows).to_csv(run_dir / "training_probes.csv", index=False)
    with (run_dir / "arm_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary


def _alpha_seq(run_dir: Path) -> list[float]:
    import pandas as pd

    return [float(x) for x in pd.read_csv(run_dir / "step_log.csv")["alpha"]]


def calibrate(budget: float, episodes: int, tags: list[str] | None = None) -> list[dict]:
    """Run the full descending ladder scan and the crossing-estimate scan on
    identical step streams and compare step for step (report §8q amendment)."""
    import tempfile

    ck_map = dict(captured_checkpoints())
    if tags is None:
        tags = sorted(ck_map)[:1] + sorted(ck_map)[-1:]  # first and last checkpoint
    results = []
    for tag in tags:
        ck_path = ck_map[tag]
        runs = {}
        for name, fast in (("full_ladder", False), ("fast_crossing", True)):
            with tempfile.TemporaryDirectory() as td:
                run_one(ck_path, tag, Path(td), budget, episodes, fast=fast)
                runs[name] = _alpha_seq(Path(td))
        a_f, a_x = runs["full_ladder"], runs["fast_crossing"]
        agree = len(a_f) == len(a_x) and all(x == y for x, y in zip(a_f, a_x))
        results.append({
            "tag": tag, "budget": float(budget), "episodes": int(episodes),
            "n_steps": len(a_f),
            "agree": bool(agree),
            "disagreements": [
                {"step": i, "full": x, "fast": y}
                for i, (x, y) in enumerate(zip(a_f, a_x)) if x != y
            ],
        })
        print(f"[cal] {tag} eps{budget}: {len(a_f)} steps, "
              f"agree={agree}, " + (f"disagreements={results[-1]['disagreements'][:4]}"
                                    if not agree else ""), flush=True)
    return results


def _merge(out_dir: Path) -> None:
    merged: dict = {}
    for f in sorted(out_dir.glob("*__*/arm_summary.json")):
        merged[f.parent.name] = json.loads(f.read_text())
    (out_dir / "summary.json").write_text(json.dumps(merged, indent=2, default=float))
    print(f"merged {len(merged)} runs -> {out_dir / 'summary.json'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/delay30_maintenance")
    ap.add_argument("--budgets", default="0.2,0.05")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--calibrate", action="store_true",
                    help="run the full-ladder vs crossing-estimate calibration "
                         "check and exit (writes calibration.json)")
    args = ap.parse_args()

    out_dir = ROOT / args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge:
        _merge(out_dir)
        return 0
    if args.calibrate:
        cal_results = []
        for budget in [float(b) for b in args.budgets.split(",")]:
            cal_results.extend(calibrate(budget, episodes=15))
        (out_dir / "calibration.json").write_text(
            json.dumps(cal_results, indent=2, default=float))
        agree_all = all(r["agree"] for r in cal_results)
        print(f"CALIBRATION: {'AGREE' if agree_all else 'DISAGREE'} "
              f"({sum(r['agree'] for r in cal_results)}/{len(cal_results)} tag/eps pairs)")
        return 0

    budgets = [float(b) for b in args.budgets.split(",")]
    ckpts = captured_checkpoints()
    ck_map = dict(ckpts)
    # shard over checkpoints; each checkpoint runs all arms
    mine = [(tag, ck) for i, (tag, ck) in enumerate(ckpts)
            if i % args.num_shards == args.shard]
    n_runs = len(mine) * 2 * len(budgets)
    print(f"[shard {args.shard}/{args.num_shards}] {len(mine)} of {len(ckpts)} "
          f"checkpoints x 2 arms x {len(budgets)} budgets = {n_runs} runs", flush=True)

    failures = 0
    for tag, _ in mine:
        ck_path = ck_map[tag]
        for budget in budgets:
            b_dir = out_dir / f"{tag}__budget_eps{budget}"
            c_dir = out_dir / f"{tag}__matched_raw_eps{budget}"
            try:
                if (b_dir / "arm_summary.json").exists():
                    print(f"[skip] {tag} budget_eps{budget} (complete)", flush=True)
                else:
                    print(f"[run ] {tag} budget_eps{budget} -> {b_dir}", flush=True)
                    s, norms = run_one(ck_path, tag, b_dir, budget, args.episodes,
                                       fast=False)
                    r, b = s["rubric"], s["battery"]
                    print(f"      delayed={b['delayed_intact']:.3f} "
                          f"follows={r['subst_diff_follows']:.3f} "
                          f"decode={s['post_analysis'].get('decode_pre_response', float('nan')):.3f} "
                          f"d_gnn={s['param_deltas']['gnn']:.3f} "
                          f"rej={s['alpha_rejected_frac']:.2f} disp_max={s['displacement_max']:.3f} "
                          f"ALL={'PASS' if r['all_criteria'] else 'fail'}", flush=True)
                    (b_dir / "accepted_norms.json").write_text(json.dumps(norms))
                if (c_dir / "arm_summary.json").exists():
                    print(f"[skip] {tag} matched_raw_eps{budget} (complete)", flush=True)
                else:
                    norm_path = b_dir / "accepted_norms.json"
                    if not norm_path.exists():
                        print(f"      [skip control] missing {norm_path}", flush=True)
                        continue
                    print(f"[run ] {tag} matched_raw_eps{budget} -> {c_dir}", flush=True)
                    s = run_control(ck_path, tag, c_dir, budget, norm_path, args.episodes)
                    r, b = s["rubric"], s["battery"]
                    print(f"      delayed={b['delayed_intact']:.3f} "
                          f"follows={r['subst_diff_follows']:.3f} "
                          f"decode={s['post_analysis'].get('decode_pre_response', float('nan')):.3f} "
                          f"d_gnn={s['param_deltas']['gnn']:.3f} "
                          f"ALL={'PASS' if r['all_criteria'] else 'fail'}", flush=True)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"      ERROR in {tag} eps{budget}: {exc}", flush=True)
                raise
    _merge(out_dir)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
