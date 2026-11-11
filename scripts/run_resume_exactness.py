"""M1-a registered experiment (TECHNICAL_REPORT §10): deterministic resume.

Phases (each a separate process invocation of this script):

  control : train 400 interleaved episodes, run the registered battery P.
  first   : train 200 episodes, then save the execution state; with
            ``--mid-episode``, instead continue into episode 200 and pause
            after INTERRUPT_STEP steps (the episode-start snapshot is captured
            by the hook fired at that episode's reset).
  second  : NEW process - resume from the saved state (or the episode-start
            snapshot with ``--mid-episode``), train the remaining 200
            episodes, run the registered battery P.

Registered battery P = {immediate_test_accuracy, delayed_test_accuracy,
delayed_intact, delayed_state_reset} on the fixed 50-instance test set.
Registered PASS per seed: P equals the control's exactly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from artificial_system.data.graph_tasks import make_connectivity_dataset
from artificial_system.environment.graph_environment import (
    GraphClassificationEnvironment,
)
from artificial_system.evaluation import graph_classification as graph_eval
from artificial_system.models.gnn_ssm import build_model
from artificial_system.training.trainer import Trainer
from artificial_system.utils.checkpoints import (
    resume_execution_state,
    save_execution_state,
)
from artificial_system.utils.seed import set_seed

# ---- registered constants (TECHNICAL_REPORT §10, fixed before any run) ----
DELAY = 3
EPISODES = 400
SPLIT_EPISODE = 200
INTERRUPT_STEP = 5           # steps into the interrupted episode (E1b)
BATCH = 16
SEQ_LEN = 32
DATASET_SEED = 123
GEN = dict(num_nodes=60, feature_dim=16, marker_dim=2, max_path_len=3,
           edge_density=0.05, directed=False, label_mode="path",
           neg_min_distance=3)

BATTERY_KEYS = ("immediate_test_accuracy", "delayed_test_accuracy",
                "delayed_intact", "delayed_state_reset")


def model_cfg() -> dict:
    return {
        "gnn": {"layers": 2, "hidden_dim": 32, "dropout": 0.0,
                "aggregation": "mean", "residual": True,
                "use_edge_features": False},
        "ssm": {"state_dim": 32, "diagonal": True, "max_radius": 0.99,
                "init_decay": 0.9, "nonlinearity": "tanh"},
        "readout": {"pooling": "mean", "head_hidden": 0},
    }


def model_dims() -> dict:
    return {"node_feature_dim": GEN["feature_dim"], "edge_feature_dim": 0,
            "num_actions": 2, "num_nodes": GEN["num_nodes"]}


def build_arm(seed: int):
    """Freshly seeded model/optimizer/trainers/environments (both phases)."""
    set_seed(seed)
    train_ds = make_connectivity_dataset(200, seed=DATASET_SEED, **GEN)
    test_ds = make_connectivity_dataset(50, seed=DATASET_SEED + 1, **GEN)
    model = build_model(model_cfg(), node_feature_dim=GEN["feature_dim"],
                        edge_feature_dim=0, num_actions=2,
                        num_nodes=GEN["num_nodes"])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    task = {"delay": DELAY, "marker_dim": GEN["marker_dim"],
            "blank_mode": "generic"}
    env_imm = GraphClassificationEnvironment(
        train_ds, {**task, "delay": 0}, seed=seed * 2000 + 1)
    env_del = GraphClassificationEnvironment(
        train_ds, task, seed=seed * 2000 + 2)
    train_cfg = {"batch_size": BATCH, "sequence_length": SEQ_LEN}
    tr_imm = Trainer(model, env_imm, optimizer, train_cfg)
    tr_del = Trainer(model, env_del, optimizer, train_cfg)
    return (model, optimizer, tr_imm, tr_del,
            {"immediate": env_imm, "delayed": env_del}, test_ds)


def run_interleaved(tr_imm, tr_del, start_ep: int, end_ep: int) -> None:
    """Condition-D interleaving on episode indices [start_ep, end_ep);
    even indices -> immediate, odd -> delayed (matches §5 machinery)."""
    ep = start_ep
    while ep < end_ep:
        if ep % 2 == 0:
            tr_imm.train(1)
        else:
            tr_del.train(1)
        ep += 1


def registered_battery(model, test_ds, env_del) -> dict:
    generic = env_del._generic_graph
    imm = graph_eval.probe_dataset(model, test_ds, delay=0, batch_size=BATCH,
                                   marker_dim=GEN["marker_dim"])
    dly = graph_eval.probe_dataset(model, test_ds, delay=DELAY,
                                   batch_size=BATCH,
                                   marker_dim=GEN["marker_dim"],
                                   generic_blank=generic)
    ivr = graph_eval.probe_delayed_intact_vs_reset(
        model, test_ds, delay=DELAY, batch_size=BATCH,
        marker_dim=GEN["marker_dim"], generic_blank=generic)
    return {"immediate_test_accuracy": imm["accuracy"],
            "delayed_test_accuracy": dly["accuracy"],
            "delayed_intact": ivr["delayed_intact"],
            "delayed_state_reset": ivr["delayed_state_reset"]}


def trainer_steps(tr_imm, tr_del) -> dict:
    return {"immediate": int(getattr(tr_imm, "_global_step", 0)),
            "delayed": int(getattr(tr_del, "_global_step", 0))}


def restore_trainers(payload, tr_imm, tr_del) -> None:
    steps = (payload.get("extra") or {}).get("trainer_steps")
    if steps:
        tr_imm.load_execution_state({"global_step": steps["immediate"]})
        tr_del.load_execution_state({"global_step": steps["delayed"]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["control", "first", "second"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mid-episode", action="store_true",
                    help="E1b: interrupt INTERRUPT_STEP steps into episode "
                         "SPLIT_EPISODE; resume by episode restart")
    ap.add_argument("--outdir", default="results/m1_resume")
    args = ap.parse_args()

    if args.phase == "control":
        tag = "E1_seed{:03d}__control".format(args.seed)
    elif args.mid_episode:
        tag = "E1b_seed{:03d}__resume_midep".format(args.seed)
    else:
        tag = "E1_seed{:03d}__resume".format(args.seed)  # first+second share it
    run_dir = Path(args.outdir) / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    model, optimizer, tr_imm, tr_del, envs, test_ds = build_arm(args.seed)
    extra = {"model_dims": model_dims()}

    if args.phase == "control":
        run_interleaved(tr_imm, tr_del, 0, EPISODES)
        payload_out = registered_battery(model, test_ds, envs["delayed"])

    elif args.phase == "first":
        if args.mid_episode:
            counter = {"episodes_seen": 0}
            snap_path = run_dir / "episode_start.pt"

            def hook(episode: int, trainer: Trainer) -> None:
                idx = counter["episodes_seen"]
                counter["episodes_seen"] += 1
                if idx == SPLIT_EPISODE:
                    # snapshot BEFORE the reset of the episode to interrupt;
                    # trainer_steps from the live trainer objects
                    save_execution_state(
                        snap_path, model, optimizer,
                        global_step=trainer._current_step, envs=envs,
                        config=model_cfg(), seed=args.seed,
                        extra={**extra,
                               "trainer_steps": trainer_steps(tr_imm, tr_del)})

            tr_imm.episode_snapshot_fn = hook
            tr_del.episode_snapshot_fn = hook
            run_interleaved(tr_imm, tr_del, 0, SPLIT_EPISODE)
            tr_imm.train(1, pause_after_steps=INTERRUPT_STEP)
            payload_out = {"interrupted": "mid-episode",
                           "episode": SPLIT_EPISODE,
                           "steps_into_episode": INTERRUPT_STEP}
        else:
            run_interleaved(tr_imm, tr_del, 0, SPLIT_EPISODE)
            save_execution_state(
                run_dir / "exec_state.pt", model, optimizer,
                global_step=SPLIT_EPISODE * SEQ_LEN, envs=envs,
                config=model_cfg(), seed=args.seed,
                extra={**extra, "trainer_steps": trainer_steps(tr_imm, tr_del)})
            payload_out = {"interrupted": "episode-boundary",
                           "episodes_done": SPLIT_EPISODE}

    else:  # second
        if args.mid_episode:
            path = run_dir / "episode_start.pt"
        else:
            path = run_dir / "exec_state.pt"
        payload = resume_execution_state(path, model, optimizer, envs=envs)
        restore_trainers(payload, tr_imm, tr_del)
        run_interleaved(tr_imm, tr_del, SPLIT_EPISODE, EPISODES)
        payload_out = registered_battery(model, test_ds, envs["delayed"])

    payload_out.update({"phase": args.phase, "seed": args.seed,
                        "mid_episode": bool(args.mid_episode)})
    with open(run_dir / "probe_summary.json", "w", encoding="utf-8") as fh:
        json.dump(payload_out, fh, indent=2)
    print(f"[{tag}] phase={args.phase} done -> {run_dir / 'probe_summary.json'}")


if __name__ == "__main__":
    main()
