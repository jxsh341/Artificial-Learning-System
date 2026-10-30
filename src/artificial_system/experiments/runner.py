"""Experiment runner (spec §22, §24, §25, §30).

``run_experiment`` is the Definition-of-Done entry point: given a config
and a graph source it loads + validates the graph, builds GNN + SSM, trains
the combined system, saves checkpoints, runs the ARRG evaluation and any
configured ablations, and exports metrics plus an aggregated summary — all
reproducible from the configuration file.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import torch
import yaml

from artificial_system.data.adapters import load_graph
from artificial_system.data.graph_tasks import make_connectivity_dataset
from artificial_system.data.schema import GraphData
from artificial_system.environment.cue_environment import CueLearningEnvironment
from artificial_system.environment.graph_environment import GraphClassificationEnvironment
from artificial_system.environment.tasks import make_cue_features
from artificial_system.evaluation import (
    acquisition as acquisition_eval,
)
from artificial_system.evaluation import (
    generalization as generalization_eval,
)
from artificial_system.evaluation import (
    graph_classification as graph_eval,
)
from artificial_system.evaluation import (
    memory_curve,
)
from artificial_system.evaluation import (
    perturbations,
)
from artificial_system.evaluation import (
    retention as retention_eval,
)
from artificial_system.evaluation import (
    reversal as reversal_eval,
)
from artificial_system.evaluation.statistics import aggregate_summaries
from artificial_system.experiments.ablations import build_ablation_model
from artificial_system.models.gnn_ssm import build_model
from artificial_system.training.trainer import Trainer
from artificial_system.utils.checkpoints import load_checkpoint, save_checkpoint
from artificial_system.utils.logging import ExperimentLogger, get_logger
from artificial_system.utils.seed import set_seed

log = get_logger("runner")

CONFIG_FILENAME = "config.yaml"


# ---------------------------------------------------------------- factories


def load_config(config_path: str | Path) -> dict:
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def make_env_factory(cfg: dict, base_graph: GraphData):
    """Create environments with shared cue structure and deterministic seeds."""
    task = dict(cfg.get("task", {}))
    cue_features, family_ids = make_cue_features(
        int(task.get("num_cues", 4)),
        int(task.get("cue_feature_dim", 8)),
        structure=task.get("structure", "onehot"),
        num_families=int(task.get("num_families", 2)),
        feature_noise=float(task.get("feature_noise", 0.25)),
        seed=int(task.get("feature_seed", 12345)),
    )
    default_allowed = task.get("train_cues")

    def factory(allowed_cues=None, seed: int = 0, task_overrides: dict | None = None) -> CueLearningEnvironment:
        return CueLearningEnvironment(
            base_graph,
            {**task, **(task_overrides or {})},
            cue_features=cue_features,
            family_ids=family_ids,
            allowed_cues=allowed_cues if allowed_cues is not None else default_allowed,
            seed=seed,
        )

    return factory, cue_features, family_ids


# ---------------------------------------------------------------- single run


def run_single(
    cfg: dict,
    base_graph: GraphData,
    seed: int,
    run_dir: Path,
    model_name: str | None = None,
) -> dict:
    """Run one seed: train -> checkpoint -> ARRG evaluation -> metrics export."""
    set_seed(seed)
    train_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})

    if model_name is None or model_name == "M4":
        model = build_model(
            cfg,
            node_feature_dim=base_graph.node_feature_dim,
            edge_feature_dim=base_graph.edge_feature_dim,
            num_actions=int(cfg.get("task", {}).get("num_actions", 2)),
            num_nodes=base_graph.num_nodes,
        )
    else:
        model = build_ablation_model(
            model_name,
            cfg,
            node_feature_dim=base_graph.node_feature_dim,
            edge_feature_dim=base_graph.edge_feature_dim,
            num_actions=int(cfg.get("task", {}).get("num_actions", 2)),
            num_nodes=base_graph.num_nodes,
        )
    n_params = sum(p.numel() for p in model.parameters())

    env_factory, _cue_feats, _fam = make_env_factory(cfg, base_graph)
    env = env_factory(seed=seed * 1000 + 1)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    logger = ExperimentLogger(run_dir)
    trainer = Trainer(model, env, optimizer, train_cfg, logger=logger)

    summary: dict = {"seed": seed, "num_parameters": n_params}
    save_enabled = bool(cfg.get("checkpointing", {}).get("save", True))

    # ---- Acquisition (§14) ----
    if eval_cfg.get("acquisition", True):
        t0 = time.time()
        records, acq = acquisition_eval.run_acquisition(
            trainer,
            episodes=int(train_cfg.get("episodes", 100)),
            criterion=float(eval_cfg.get("criterion", 0.9)),
            criterion_window=int(eval_cfg.get("criterion_window", 25)),
        )
        logger.write_table("metrics", records)
        summary.update({f"acquisition_{k}": v for k, v in acq.items()})
        summary["train_seconds"] = round(time.time() - t0, 2)

    if save_enabled:
        save_checkpoint(
            run_dir / "checkpoint.pt",
            model,
            optimizer,
            step=summary.get("acquisition_steps", 0),
            config=cfg,
            seed=seed,
            env_config=cfg.get("task", {}),
        )

    # ---- Retention (§15) ----
    if eval_cfg.get("retention", True):
        eval_env = env_factory(seed=seed * 1000 + 2)
        eval_env.reset(int(eval_cfg.get("eval_batch_size", 16)))
        rows, ret = retention_eval.run_retention(
            model,
            eval_env,
            delays=list(eval_cfg.get("retention_delays", [0, 10, 50, 100, 500])),
            test_steps=int(eval_cfg.get("retention_test_steps", 100)),
            distractor=eval_cfg.get("retention_distractor", "blank"),
        )
        logger.write_table("retention", rows)
        summary.update(ret)

    # ---- Generalization (§17) ----
    task_cfg = cfg.get("task", {})
    if eval_cfg.get("generalization", True) and task_cfg.get("test_cues"):
        rows, gen = generalization_eval.run_generalization(
            model,
            env_factory,
            train_cues=list(task_cfg.get("train_cues") or range(task_cfg.get("num_cues", 4))),
            test_cues=list(task_cfg["test_cues"]),
            test_steps=int(eval_cfg.get("generalization_test_steps", 200)),
            seed=seed * 1000 + 3,
            batch_size=int(eval_cfg.get("eval_batch_size", 16)),
        )
        logger.write_table("generalization", rows)
        summary.update(gen)

    # ---- Extended analyses (phase 2-4) ----
    eval_batch = int(eval_cfg.get("eval_batch_size", 16))

    def build_env(graph: GraphData, s: int, task_overrides: dict | None = None) -> CueLearningEnvironment:
        factory, _, _ = make_env_factory(cfg, graph)
        return factory(seed=s, task_overrides=task_overrides)

    if eval_cfg.get("memory_curve", False):
        eval_env_mc = env_factory(seed=seed * 1000 + 4)
        eval_env_mc.reset(eval_batch)
        rows, mc = memory_curve.run_memory_curve(
            model,
            eval_env_mc,
            delays=list(eval_cfg.get("memory_curve_delays", [0, 2, 10, 50, 100, 500, 1000, 5000])),
            repeats=int(eval_cfg.get("memory_curve_repeats", 20)),
            distractor=eval_cfg.get("retention_distractor", "blank"),
        )
        logger.write_table("memory_curve", rows)
        summary.update(mc)

    if eval_cfg.get("temporal_perturbations", False):
        eval_env_tp = env_factory(seed=seed * 1000 + 5)
        eval_env_tp.reset(eval_batch)
        rows, tp = perturbations.run_temporal_perturbations(
            model,
            eval_env_tp,
            num_steps=int(eval_cfg.get("perturbation_steps", 200)),
            batch_size=eval_batch,
            noise_sigmas=eval_cfg.get("noise_sigmas", [0.1, 0.5, 1.0]),
            reset_fractions=eval_cfg.get("reset_fractions", [0.25, 0.5, 0.75]),
            truncation_intervals=eval_cfg.get("truncation_intervals", [2, 5, 10]),
        )
        logger.write_table("temporal_perturbations", rows)
        summary.update({f"temporal_{k}": v for k, v in tp.items()})

    if eval_cfg.get("storage_locus", False):
        rows, sl = perturbations.storage_locus(
            model,
            build_env,
            base_graph,
            num_steps=int(eval_cfg.get("perturbation_steps", 200)),
            batch_size=eval_batch,
            seed=seed * 1000 + 6,
        )
        logger.write_table("storage_locus", rows)
        summary.update(
            {(k if k.startswith("locus_") else f"locus_{k}"): v for k, v in sl.items()}
        )

    if eval_cfg.get("graph_perturbations", False):
        rows, gp = perturbations.run_graph_perturbations(
            model,
            build_env,
            base_graph,
            probe_steps=int(eval_cfg.get("perturbation_steps", 200)),
            batch_size=eval_batch,
            seed=seed * 1000 + 7,
            edge_levels=eval_cfg.get("graph_edge_levels", [0.25, 0.5, 0.75]),
            node_levels=eval_cfg.get("graph_node_levels", [0.1, 0.25, 0.5]),
            weight_levels=eval_cfg.get("graph_weight_levels", [0.25, 0.5, 1.0]),
        )
        logger.write_table("graph_perturbations", rows)
        summary.update(
            {(k if k.startswith("graph_") else f"graph_{k}"): v for k, v in gp.items()}
        )

    # ---- Reversal (§16) ----
    if eval_cfg.get("reversal", True):
        records, rev = reversal_eval.run_reversal(
            trainer,
            episodes=int(eval_cfg.get("reversal_episodes", 80)),
            criterion=float(eval_cfg.get("criterion", 0.9)),
            criterion_window=int(eval_cfg.get("criterion_window", 25)),
        )
        logger.write_table("reversal", records)
        summary.update({f"reversal_{k}": v for k, v in rev.items()})
        if save_enabled:
            save_checkpoint(
                run_dir / "checkpoint_reversed.pt",
                model,
                optimizer,
                step=rev.get("reversal_steps", 0),
                config=cfg,
                seed=seed,
                env_config=cfg.get("task", {}),
            )

    logger.save_json("seed_summary", summary)
    logger.close()
    return summary


def _write_summary(exp_dir: Path, aggregate: dict) -> Path:
    """Write the aggregated experiment summary."""
    from artificial_system.utils.logging import _json_default

    path = exp_dir / "summary.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(aggregate, fh, indent=2, default=_json_default)
    return path


# ---------------------------------------------------------------- graph task


def _merge_seed_summaries(exp_dir: Path, summaries: list[dict]) -> list[dict]:
    """Merge freshly computed summaries with previously completed seeds."""
    by_seed = {int(s["seed"]): s for s in summaries}
    for seed_dir in sorted(exp_dir.glob("seed_*")):
        path = seed_dir / "seed_summary.json"
        if not path.exists():
            continue
        try:
            existing = json.loads(path.read_text())
        except Exception:
            continue
        sid = int(existing.get("seed", -1))
        if sid not in by_seed:
            by_seed[sid] = existing
            log.info(f"merged existing seed={sid} from {seed_dir}")
    return [by_seed[k] for k in sorted(by_seed)]


def run_graph_single(
    cfg: dict,
    datasets: dict,
    seed: int,
    run_dir: Path,
    model_name: str | None = None,
) -> dict:
    """One seed of the graph-dependent benchmark (immediate or delayed)."""
    set_seed(seed)
    task_cfg = cfg.get("task", {})
    eval_cfg = cfg.get("evaluation", {})
    train_cfg = cfg.get("training", {})
    train_ds, test_ds = datasets["train"], datasets["test"]
    unseen_ds = datasets.get("unseen_size")
    eval_batch = int(eval_cfg.get("eval_batch_size", 16))
    delay = int(task_cfg.get("delay", 0))
    marker_dim = int(task_cfg.get("marker_dim", 2))

    if model_name is None or model_name == "M4":
        model = build_model(
            cfg,
            node_feature_dim=train_ds.graphs[0].node_feature_dim,
            edge_feature_dim=train_ds.graphs[0].edge_feature_dim,
            num_actions=2,
            num_nodes=None,
        )
    else:
        model = build_ablation_model(
            model_name,
            cfg,
            node_feature_dim=train_ds.graphs[0].node_feature_dim,
            edge_feature_dim=train_ds.graphs[0].edge_feature_dim,
            num_actions=2,
            num_nodes=None,
        )
    n_params = sum(p.numel() for p in model.parameters())

    env = GraphClassificationEnvironment(train_ds, task_cfg, seed=seed * 1000 + 1)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.get("training", {}).get("learning_rate", 1e-3)),
        weight_decay=float(cfg.get("training", {}).get("weight_decay", 0.0)),
    )
    logger = ExperimentLogger(run_dir)
    trainer = Trainer(model, env, optimizer, cfg.get("training", {}), logger=logger)

    summary: dict = {"seed": seed, "num_parameters": n_params}
    save_enabled = bool(cfg.get("checkpointing", {}).get("save", True))

    # ---- Acquisition (training; protocol §14; optional staged curriculum) ----
    t0 = time.time()
    curriculum = train_cfg.get("curriculum")
    if delay > 0 and curriculum:
        # Phase 1: immediate task — the GNN + head learn the relational rule
        # with no memory pressure.
        env0 = GraphClassificationEnvironment(
            train_ds, {**task_cfg, "delay": 0}, seed=seed * 1000 + 11
        )
        trainer0 = Trainer(model, env0, optimizer, train_cfg, logger=logger)
        rec1, acq1 = acquisition_eval.run_acquisition(
            trainer0,
            episodes=int(curriculum.get("immediate_episodes", 200)),
            criterion=float(eval_cfg.get("criterion", 0.9)),
            criterion_window=int(eval_cfg.get("criterion_window", 25)),
        )
        logger.write_table("metrics_phase1_immediate", rec1)
        summary["curriculum_immediate_accuracy"] = acq1["accuracy_last_10pct"]
        log.info(
            f"curriculum phase 1 (immediate): acc={acq1['accuracy_last_10pct']:.3f}"
        )
        # Phase 2: delayed task — only the state-bridge remains to learn
        # (memorization is infeasible: generic blanks + bottlenecked state).
        records, acq = acquisition_eval.run_acquisition(
            trainer,
            episodes=int(curriculum.get("delayed_episodes", 400)),
            criterion=float(eval_cfg.get("criterion", 0.9)),
            criterion_window=int(eval_cfg.get("criterion_window", 25)),
        )
        logger.write_table("metrics", records)
        summary.update({f"acquisition_{k}": v for k, v in acq.items()})
    else:
        records, acq = acquisition_eval.run_acquisition(
            trainer,
            episodes=int(train_cfg.get("episodes", 100)),
            criterion=float(eval_cfg.get("criterion", 0.9)),
            criterion_window=int(eval_cfg.get("criterion_window", 25)),
        )
        logger.write_table("metrics", records)
        summary.update({f"acquisition_{k}": v for k, v in acq.items()})
    summary["train_seconds"] = round(time.time() - t0, 2)
    if save_enabled:
        save_checkpoint(
            run_dir / "checkpoint.pt", model, optimizer,
            step=summary.get("acquisition_steps", 0), config=cfg, seed=seed,
            env_config=task_cfg,
        )

    # ---- Task-consistent probe on unseen instances (+ failure-mode view) ----
    generic_blank = (
        env._generic_graph
        if delay > 0 and getattr(env, "blank_mode", "no_markers") == "generic"
        else None
    )
    probe = graph_eval.probe_dataset(
        model, test_ds, delay=delay, batch_size=eval_batch,
        marker_dim=marker_dim, generic_blank=generic_blank,
    )
    eval_rows = [{"condition": "test_task", **{k: v for k, v in probe.items() if k != "accuracy_by_path_len"}}]
    for pl, acc in probe["accuracy_by_path_len"].items():
        eval_rows.append({"condition": f"test_path_len_{pl}", "accuracy": acc, "path_len": int(pl)})
    logger.write_table("graph_eval", eval_rows)
    summary["graph_task_accuracy"] = probe["accuracy"]
    summary["graph_task_accuracy_by_class"] = probe["accuracy_by_class"]
    summary["graph_task_accuracy_by_path_len"] = probe["accuracy_by_path_len"]

    # ---- M7 rewired control: does the prediction depend on correct edges? ----
    rows, rw = graph_eval.run_rewired_control(
        model, test_ds,
        levels=list(eval_cfg.get("rewired_levels", [0.25, 0.5, 1.0])),
        batch_size=eval_batch, seed=seed * 1000 + 2,
    )
    logger.write_table("rewired_control", rows)
    summary.update({f"m7_{k}": v for k, v in rw.items()})

    # ---- Locus dissociation (delayed relational variant only) ----
    if delay > 0:
        rows, sl = graph_eval.run_graph_locus(
            model, test_ds, delay=delay, batch_size=eval_batch,
            marker_dim=marker_dim, generic_blank=generic_blank,
        )
        logger.write_table("graph_locus", rows)
        summary.update(
            {(k if k.startswith("locus_") else f"locus_{k}"): v for k, v in sl.items()}
        )

    # ---- Generalization: unseen instances + unseen size ----
    rows, gen = graph_eval.run_graph_generalization(
        model, test_ds, unseen_size_dataset=unseen_ds, delay=delay, batch_size=eval_batch,
        generic_blank=generic_blank,  # must match the trained blank protocol
    )
    logger.write_table("graph_generalization", rows)
    summary.update(gen)

    logger.save_json("seed_summary", summary)
    logger.close()
    return summary


def run_graph_experiment(
    cfg: dict,
    exp_dir: Path,
    seed_list: list[int],
) -> dict:
    """Graph-dependent benchmark pipeline (immediate + delayed variants)."""
    task_cfg = cfg.get("task", {})
    gen_cfg = dict(
        num_nodes=int(task_cfg.get("num_nodes", 100)),
        feature_dim=int(cfg.get("graph", {}).get("node_feature_dim", 16)),
        marker_dim=int(task_cfg.get("marker_dim", 2)),
        max_path_len=int(task_cfg.get("max_path_len", 3)),
        edge_density=float(task_cfg.get("edge_density", 0.05)),
        directed=bool(task_cfg.get("directed", False)),
        label_mode=task_cfg.get("label_mode", "path"),
        neg_min_distance=int(task_cfg.get("neg_min_distance", 3)),
    )
    ds_seed = int(task_cfg.get("dataset_seed", 123))
    train_ds = make_connectivity_dataset(int(task_cfg.get("num_train", 400)), seed=ds_seed, **gen_cfg)
    test_ds = make_connectivity_dataset(int(task_cfg.get("num_test", 100)), seed=ds_seed + 1, **gen_cfg)
    unseen_ds = None
    if task_cfg.get("test_num_nodes"):
        unseen_ds = make_connectivity_dataset(
            int(task_cfg.get("num_test", 100)),
            seed=ds_seed + 2,
            **{**gen_cfg, "num_nodes": int(task_cfg["test_num_nodes"])},
        )
    log.info(
        f"graph task: {len(train_ds)} train / {len(test_ds)} test instances, "
        f"n={gen_cfg['num_nodes']}, max_path_len={gen_cfg['max_path_len']}, "
        f"delay={task_cfg.get('delay', 0)}"
    )
    log.info(f"dataset stats: {train_ds.metadata['stats']}")
    datasets = {"train": train_ds, "test": test_ds, "unseen_size": unseen_ds}

    summaries = []
    for seed in seed_list:
        seed_dir = exp_dir / f"seed_{seed:03d}"
        log.info(f"[{cfg.get('experiment_name')}] seed={seed} -> {seed_dir}")
        summaries.append(run_graph_single(cfg, datasets, seed, seed_dir))
    summaries = _merge_seed_summaries(exp_dir, summaries)
    aggregate = aggregate_summaries(summaries)
    aggregate["per_seed"] = summaries

    abl_cfg = cfg.get("ablations", {})
    if abl_cfg.get("enabled", False):
        # M1..M6 model variants (M6 = message passing disabled = M8 control);
        # M7 is an evaluation-level control (rewired probes above).
        for name in abl_cfg.get("models", ["M1", "M2", "M3", "M4", "M6"]):
            abl_summaries = []
            for seed in seed_list:
                seed_dir = exp_dir / "ablations" / name / f"seed_{seed:03d}"
                log.info(f"[{cfg.get('experiment_name')}] ablation {name} seed={seed}")
                abl_summaries.append(
                    run_graph_single(cfg, datasets, seed, seed_dir, model_name=name)
                )
            aggregate[f"ablation_{name}"] = aggregate_summaries(abl_summaries)
            aggregate[f"ablation_{name}_per_seed"] = abl_summaries
    return aggregate


# ---------------------------------------------------------------- curriculum study


DEFAULT_CONDITIONS = {
    "A_immediate": {"schedule": [["immediate", 400]]},
    "B_delayed": {"schedule": [["delayed", 400]]},
    "C_staged": {"schedule": [["immediate", 200], ["delayed", 200]]},
    "D_interleaved": {"schedule": "interleaved"},
}


def run_curriculum_condition(
    cfg: dict,
    datasets: dict,
    seed: int,
    run_dir: Path,
    cond_name: str,
    schedule,
    study_cfg: dict,
    episodes_total_override: int | None = None,
) -> dict:
    """One seed of one curriculum condition.

    Chunked, budget-matched training with intact-vs-reset probes throughout
    (Phase C), then the full evaluation battery (Phase D).
    """
    set_seed(seed)
    task_cfg = dict(cfg.get("task", {}))
    train_cfg = dict(cfg.get("training", {}))
    eval_cfg = cfg.get("evaluation", {})
    train_ds, test_ds = datasets["train"], datasets["test"]
    unseen_ds = datasets.get("unseen_size")
    delay = int(task_cfg.get("delay", 3))
    marker_dim = int(task_cfg.get("marker_dim", 2))
    seq_len = int(train_cfg.get("sequence_length", 32))
    eval_batch = int(eval_cfg.get("eval_batch_size", 16))

    model = build_model(
        cfg,
        node_feature_dim=train_ds.graphs[0].node_feature_dim,
        edge_feature_dim=train_ds.graphs[0].edge_feature_dim,
        num_actions=2,
        num_nodes=None,
    )
    n_params = sum(p.numel() for p in model.parameters())

    env_imm = GraphClassificationEnvironment(train_ds, {**task_cfg, "delay": 0}, seed=seed * 2000 + 1)
    env_del = GraphClassificationEnvironment(train_ds, task_cfg, seed=seed * 2000 + 2)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    logger = ExperimentLogger(run_dir)
    trainer_imm = Trainer(model, env_imm, optimizer, train_cfg, logger=logger)
    trainer_del = Trainer(model, env_del, optimizer, train_cfg, logger=logger)
    generic_blank = (
        env_del._generic_graph
        if getattr(env_del, "blank_mode", "no_markers") == "generic"
        else None
    )

    total = int(
        episodes_total_override
        if episodes_total_override is not None
        else study_cfg.get("episodes_total", 400)
    )
    chunk = max(1, int(study_cfg.get("chunk_size", 100)))

    probe_rows: list[dict] = []
    episodes_done = 0

    def set_phase_trainable(phase: str) -> None:
        """``delayed_frozen`` protects the learned relational representation by
        freezing the GNN during bridge training (single-variable intervention)."""
        frozen = phase == "delayed_frozen"
        for p in model.graph_encoder.parameters():
            p.requires_grad_(not frozen)

    def run_probe(tag: str) -> None:
        imm = graph_eval.probe_dataset(
            model, test_ds, delay=0, batch_size=eval_batch, marker_dim=marker_dim
        )
        dv = graph_eval.probe_delayed_intact_vs_reset(
            model, test_ds, delay=delay, batch_size=eval_batch,
            marker_dim=marker_dim, generic_blank=generic_blank,
        )
        probe_rows.append({"tag": tag, "episodes_done": episodes_done,
                           "immediate_test": imm["accuracy"], **dv})
        logger.log_scalar(f"{cond_name}/delayed_intact", dv["delayed_intact"], episodes_done)
        logger.log_scalar(f"{cond_name}/delayed_reset", dv["delayed_state_reset"], episodes_done)

    if schedule == "interleaved":
        while episodes_done < total:
            trainer_imm.train(1)
            episodes_done += 1
            if episodes_done < total:
                trainer_del.train(1)
                episodes_done += 1
            if episodes_done % chunk == 0 or episodes_done >= total:
                run_probe("interleaved")
    else:
        norm_schedule = [(ph, int(ep)) for ph, ep in schedule]
        for phase, episodes in norm_schedule:
            set_phase_trainable(phase)
            trainer = trainer_imm if phase == "immediate" else trainer_del
            done_in_phase = 0
            while done_in_phase < episodes:
                n = min(chunk, episodes - done_in_phase)
                trainer.train(n)
                done_in_phase += n
                episodes_done += n
                run_probe(phase)
        set_phase_trainable("immediate")  # restore for any post-training use

    logger.write_table("training_probes", probe_rows)

    # ---------------------------------------------------------------- battery
    summary: dict = {
        "seed": seed,
        "condition": cond_name,
        "num_parameters": n_params,
        "episodes_total": episodes_done,
    }
    # Examples-seen report (steps are budget-matched; presentation counts
    # differ by construction because delay steps consume steps).
    if schedule == "interleaved":
        n_imm, n_del = total // 2, total - total // 2
    else:
        n_imm = sum(ep for ph, ep in [(ph, int(ep)) for ph, ep in schedule] if ph == "immediate")
        n_del = episodes_done - n_imm
    summary["presentations_immediate"] = n_imm * seq_len
    summary["presentations_delayed"] = int(n_del * seq_len / max(delay + 1, 1))

    probe_imm = graph_eval.probe_dataset(
        model, test_ds, delay=0, batch_size=eval_batch, marker_dim=marker_dim
    )
    summary["immediate_test_accuracy"] = probe_imm["accuracy"]
    probe_del = graph_eval.probe_dataset(
        model, test_ds, delay=delay, batch_size=eval_batch,
        marker_dim=marker_dim, generic_blank=generic_blank,
    )
    summary["delayed_test_accuracy"] = probe_del["accuracy"]

    ivr = graph_eval.probe_delayed_intact_vs_reset(
        model, test_ds, delay=delay, batch_size=eval_batch,
        marker_dim=marker_dim, generic_blank=generic_blank,
    )
    summary.update(ivr)

    rows, ret = graph_eval.run_delay_retention(
        model, test_ds,
        delays=list(study_cfg.get("delay_retention_delays", [0, 3, 10, 30, 100])),
        batch_size=eval_batch, marker_dim=marker_dim, generic_blank=generic_blank,
    )
    logger.write_table("delay_retention", rows)
    summary.update(ret)

    rows, sl = graph_eval.run_graph_locus(
        model, test_ds, delay=delay, batch_size=eval_batch,
        marker_dim=marker_dim, generic_blank=generic_blank,
    )
    logger.write_table("graph_locus", rows)
    summary.update(sl)

    if study_cfg.get("counterfactual", True):
        rows, cf = graph_eval.run_counterfactual_probe(
            model, test_ds, delay=delay, batch_size=eval_batch,
            marker_dim=marker_dim, generic_blank=generic_blank, seed=seed * 2000 + 3,
        )
        logger.write_table("counterfactual", rows)
        summary.update(cf)

    rows, gen = graph_eval.run_graph_generalization(
        model, test_ds, unseen_size_dataset=unseen_ds, delay=delay, batch_size=eval_batch,
        generic_blank=generic_blank,  # must match the trained blank protocol
    )
    logger.write_table("graph_generalization", rows)
    summary.update(gen)

    # M7 controls, reduced levels for the study
    rows, rw = graph_eval.run_rewired_control(
        model, test_ds,
        levels=list(study_cfg.get("rewired_levels", [1.0])),
        batch_size=eval_batch, seed=seed * 2000 + 4,
    )
    logger.write_table("rewired_control", rows)
    summary.update({f"m7_{k}": v for k, v in rw.items()})

    if bool(cfg.get("checkpointing", {}).get("save", True)):
        save_checkpoint(
            run_dir / "checkpoint.pt", model, optimizer,
            step=episodes_done, config=cfg, seed=seed, env_config=task_cfg,
        )
    logger.save_json("seed_summary", summary)
    logger.close()
    return summary


def run_curriculum_study(
    config_path: str | Path | None = None,
    config: dict | None = None,
    seeds: list[int] | None = None,
) -> Path:
    """Curriculum-scheduling study (Phase B): four budget-matched conditions,
    multiple seeds, success-rate aggregation against a pre-registered
    threshold. Returns the results dir."""
    if config is None:
        if config_path is None:
            raise ValueError("provide either config or config_path")
        config = load_config(config_path)
    cfg = copy.deepcopy(config)

    exp_dir = Path(cfg.get("results_dir", "results")) / str(cfg.get("experiment_name", "curriculum_study"))
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / CONFIG_FILENAME, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    seed_list = [int(s) for s in (seeds if seeds is not None else cfg.get("seeds", [0]))]
    study_cfg = cfg.get("study", {})

    # datasets (frozen baseline generation)
    task_cfg = cfg.get("task", {})
    gen_cfg = dict(
        num_nodes=int(task_cfg.get("num_nodes", 100)),
        feature_dim=int(cfg.get("graph", {}).get("node_feature_dim", 16)),
        marker_dim=int(task_cfg.get("marker_dim", 2)),
        max_path_len=int(task_cfg.get("max_path_len", 3)),
        edge_density=float(task_cfg.get("edge_density", 0.05)),
        directed=bool(task_cfg.get("directed", False)),
        label_mode=task_cfg.get("label_mode", "adjacency"),
        neg_min_distance=int(task_cfg.get("neg_min_distance", 3)),
    )
    ds_seed = int(task_cfg.get("dataset_seed", 123))
    train_ds = make_connectivity_dataset(int(task_cfg.get("num_train", 2000)), seed=ds_seed, **gen_cfg)
    test_ds = make_connectivity_dataset(int(task_cfg.get("num_test", 200)), seed=ds_seed + 1, **gen_cfg)
    unseen_ds = None
    if task_cfg.get("test_num_nodes"):
        unseen_ds = make_connectivity_dataset(
            int(task_cfg.get("num_test", 200)),
            seed=ds_seed + 2,
            **{**gen_cfg, "num_nodes": int(task_cfg["test_num_nodes"])},
        )
    datasets = {"train": train_ds, "test": test_ds, "unseen_size": unseen_ds}
    log.info(
        f"curriculum study: {len(train_ds)}/{len(test_ds)} instances, "
        f"n={gen_cfg['num_nodes']}, delay={task_cfg.get('delay', 3)}, "
        f"{len(seed_list)} seeds x {len(study_cfg.get('conditions', DEFAULT_CONDITIONS))} conditions"
    )

    conditions = study_cfg.get("conditions", DEFAULT_CONDITIONS)
    threshold = float(study_cfg.get("success_threshold", 0.9))
    results: dict = {}
    for cond_name, spec in conditions.items():
        cond_summaries = []
        for seed in seed_list:
            run_dir = exp_dir / "curriculum" / cond_name / f"seed_{seed:03d}"
            # resume guard: skip condition-seeds that already completed
            existing_path = run_dir / "seed_summary.json"
            if existing_path.exists():
                try:
                    existing = json.loads(existing_path.read_text())
                    if int(existing.get("seed", -1)) == seed and existing.get("condition") == cond_name:
                        log.info(f"[{cond_name}] seed={seed} already complete; loading {existing_path}")
                        cond_summaries.append(existing)
                        continue
                except Exception:
                    pass
            log.info(f"[{cond_name}] seed={seed} -> {run_dir}")
            cond_summaries.append(
                run_curriculum_condition(
                    cfg, datasets, seed, run_dir, cond_name, spec.get("schedule"), study_cfg,
                    episodes_total_override=spec.get("episodes_total"),
                )
            )
        # resumable merge for condition dirs
        by_seed = {int(s["seed"]): s for s in cond_summaries}
        for path in sorted(exp_dir.glob(f"curriculum/{cond_name}/seed_*/seed_summary.json")):
            try:
                existing = json.loads(path.read_text())
            except Exception:
                continue
            sid = int(existing.get("seed", -1))
            if sid not in by_seed:
                by_seed[sid] = existing
        cond_summaries = [by_seed[k] for k in sorted(by_seed)]

        agg = aggregate_summaries(cond_summaries)
        successes = [s for s in cond_summaries if s.get("delayed_test_accuracy", 0.0) >= threshold]
        agg["success_rate"] = len(successes) / max(len(cond_summaries), 1)
        agg["successful_seeds"] = [s["seed"] for s in successes]
        agg["per_seed"] = cond_summaries
        results[cond_name] = agg

    _write_summary(exp_dir, results)
    log.info(f"curriculum study complete -> {exp_dir}")
    return exp_dir


# ---------------------------------------------------------------- experiment


def run_experiment(
    config_path: str | Path | None = None,
    config: dict | None = None,
    graph_source: str | None = None,
    seeds: list[int] | None = None,
) -> Path:
    """Full pipeline (Definition of Done, spec §30). Returns the results dir."""
    if config is None:
        if config_path is None:
            raise ValueError("provide either config or config_path")
        config = load_config(config_path)
    cfg = copy.deepcopy(config)

    exp_dir = Path(cfg.get("results_dir", "results")) / str(cfg.get("experiment_name", "experiment"))
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / CONFIG_FILENAME, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    seed_list = [int(s) for s in (seeds if seeds is not None else cfg.get("seeds", [0]))]

    task_type = str(cfg.get("task", {}).get("type", "cue"))
    if task_type == "graph_classification":
        aggregate = run_graph_experiment(cfg, exp_dir, seed_list)
        _write_summary(exp_dir, aggregate)
        log.info(f"experiment complete -> {exp_dir}")
        return exp_dir

    graph_cfg = dict(cfg.get("graph", {}))
    source = graph_source or graph_cfg.get("family", "random")
    base_graph = load_graph(source, graph_cfg)  # built + validated first (§30 step 1-2)
    cfg["graph"]["family"] = source
    log.info(
        f"graph: {base_graph.num_nodes} nodes, {base_graph.num_edges} edges, "
        f"node_dim={base_graph.node_feature_dim}, edge_dim={base_graph.edge_feature_dim}"
    )

    # ---- main model (M4) ----
    summaries = []
    for seed in seed_list:
        seed_dir = exp_dir / f"seed_{seed:03d}"
        log.info(f"[{cfg.get('experiment_name')}] seed={seed} -> {seed_dir}")
        summaries.append(run_single(cfg, base_graph, seed, seed_dir))

    # ---- merge with previously completed seeds (resumable runs) ----
    summaries = _merge_seed_summaries(exp_dir, summaries)
    aggregate = aggregate_summaries(summaries)
    aggregate["per_seed"] = summaries

    # ---- ablation matrix (§20) ----
    abl_cfg = cfg.get("ablations", {})
    if abl_cfg.get("enabled", False):
        for name in abl_cfg.get("models", ["M1", "M2", "M3", "M4", "M6"]):
            abl_summaries = []
            for seed in seed_list:
                seed_dir = exp_dir / "ablations" / name / f"seed_{seed:03d}"
                log.info(f"[{cfg.get('experiment_name')}] ablation {name} seed={seed}")
                abl_summaries.append(run_single(cfg, base_graph, seed, seed_dir, model_name=name))
            aggregate[f"ablation_{name}"] = aggregate_summaries(abl_summaries)
            aggregate[f"ablation_{name}_per_seed"] = abl_summaries

    summary_path = _write_summary(exp_dir, aggregate)
    log.info(f"experiment complete -> {exp_dir} (summary: {summary_path})")
    return exp_dir


def run_training_only(
    config_path: str | Path,
    graph_source: str | None = None,
    seeds: list[int] | None = None,
) -> Path:
    """Train-only entry point used by ``scripts/train.py``."""
    cfg = load_config(config_path)
    graph_cfg = dict(cfg.get("graph", {}))
    source = graph_source or graph_cfg.get("family", "random")
    base_graph = load_graph(source, graph_cfg)
    train_cfg = cfg.get("training", {})

    exp_dir = Path(cfg.get("results_dir", "results")) / f"{cfg.get('experiment_name', 'run')}_train"
    seed_list = [int(s) for s in (seeds if seeds is not None else cfg.get("seeds", [0]))]
    for seed in seed_list:
        set_seed(seed)
        run_dir = exp_dir / f"seed_{seed:03d}"
        model = build_model(
            cfg,
            node_feature_dim=base_graph.node_feature_dim,
            edge_feature_dim=base_graph.edge_feature_dim,
            num_actions=int(cfg.get("task", {}).get("num_actions", 2)),
            num_nodes=base_graph.num_nodes,
        )
        env_factory, _, _ = make_env_factory(cfg, base_graph)
        env = env_factory(seed=seed * 1000 + 1)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(train_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )
        logger = ExperimentLogger(run_dir)
        trainer = Trainer(model, env, optimizer, train_cfg, logger=logger)
        records, summary = acquisition_eval.run_acquisition(
            trainer, episodes=int(train_cfg.get("episodes", 100))
        )
        logger.write_table("metrics", records)
        save_checkpoint(
            run_dir / "checkpoint.pt",
            model,
            optimizer,
            step=summary.get("steps", 0),
            config=cfg,
            seed=seed,
            env_config=cfg.get("task", {}),
        )
        logger.save_json("seed_summary", summary)
        logger.close()
        log.info(f"trained seed={seed}: {summary}")
    return exp_dir


def run_evaluation_only(
    config_path: str | Path,
    checkpoint_path: str | Path,
    graph_source: str | None = None,
    seeds: list[int] | None = None,
) -> Path:
    """Evaluate a saved checkpoint without further acquisition training."""
    ckpt = load_checkpoint(checkpoint_path)
    cfg = ckpt.get("config") or load_config(config_path)
    ckpt_seed = int(ckpt.get("seed", 0))
    graph_cfg = dict(cfg.get("graph", {}))
    source = graph_source or graph_cfg.get("family", "random")
    base_graph = load_graph(source, graph_cfg)
    eval_cfg = cfg.get("evaluation", {})

    exp_dir = Path(cfg.get("results_dir", "results")) / f"{cfg.get('experiment_name', 'run')}_eval"
    run_dir = exp_dir / f"seed_{ckpt_seed:03d}"
    set_seed(ckpt_seed * 1000 + 7)
    model = build_model(
        cfg,
        node_feature_dim=base_graph.node_feature_dim,
        edge_feature_dim=base_graph.edge_feature_dim,
        num_actions=int(cfg.get("task", {}).get("num_actions", 2)),
        num_nodes=base_graph.num_nodes,
    )
    load_checkpoint(checkpoint_path, model=model)
    env_factory, _, _ = make_env_factory(cfg, base_graph)
    logger = ExperimentLogger(run_dir)

    env = env_factory(seed=ckpt_seed * 1000 + 2)
    env.reset(int(eval_cfg.get("eval_batch_size", 16)))
    rows, ret = retention_eval.run_retention(
        model,
        env,
        delays=list(eval_cfg.get("retention_delays", [0, 10, 50, 100, 500])),
        test_steps=int(eval_cfg.get("retention_test_steps", 100)),
        distractor=eval_cfg.get("retention_distractor", "blank"),
    )
    logger.write_table("retention", rows)

    task_cfg = cfg.get("task", {})
    summary = dict(ret)
    if task_cfg.get("test_cues"):
        rows, gen = generalization_eval.run_generalization(
            model,
            env_factory,
            train_cues=list(task_cfg.get("train_cues") or range(task_cfg.get("num_cues", 4))),
            test_cues=list(task_cfg["test_cues"]),
            test_steps=int(eval_cfg.get("generalization_test_steps", 200)),
            seed=ckpt_seed * 1000 + 3,
            batch_size=int(eval_cfg.get("eval_batch_size", 16)),
        )
        logger.write_table("generalization", rows)
        summary.update(gen)

    logger.save_json("eval_summary", summary)
    logger.close()
    log.info(f"evaluation complete -> {run_dir}")
    return exp_dir
