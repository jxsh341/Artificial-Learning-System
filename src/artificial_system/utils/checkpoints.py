"""Checkpointing with full reproducibility metadata (spec §24, §25)."""
from __future__ import annotations

import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def collect_env_info() -> dict:
    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        info["git_commit"] = commit.stdout.strip() if commit.returncode == 0 else None
    except Exception:
        info["git_commit"] = None
    return info


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    step: int = 0,
    config: dict | None = None,
    seed: int = 0,
    env_config: dict | None = None,
    extra: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "seed": seed,
        "config": config,
        "env_config": env_config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "env_info": collect_env_info(),
        "extra": extra or {},
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    map_location: str | torch.device = "cpu",
) -> dict:
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt


# --------------------------------------------------------- execution state (M1-a)
#
# Registered contract (TECHNICAL_REPORT §10): an execution state captures
# everything needed to continue a run exactly — parameters, optimizer,
# scheduler, global RNG streams, per-environment stream positions, and the
# trainer's global step counter. Mid-episode interruption is recovered by
# episode restart (snapshot taken at the interrupted episode's reset), because
# partial episode losses are live autograd tensors that cannot be serialized.

def capture_rng_states() -> dict:
    """Snapshot the three global RNG streams (python / numpy / torch)."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def restore_rng_states(states: dict) -> None:
    """Restore global RNG streams captured by :func:`capture_rng_states`."""
    if "python" in states and states["python"] is not None:
        random.setstate(states["python"])
    if "numpy" in states and states["numpy"] is not None:
        np.random.set_state(states["numpy"])
    if "torch" in states and states["torch"] is not None:
        torch.set_rng_state(states["torch"])


def save_execution_state(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    *,
    global_step: int = 0,
    envs: dict | None = None,
    config: dict | None = None,
    seed: int = 0,
    extra: dict | None = None,
) -> Path:
    """Save a fully resumable execution state (model + optimizer + RNGs +
    environment stream positions + global step). Environments must expose
    ``state_dict()`` / ``load_state_dict()`` (the concrete ALS environments do).

    ``extra`` should carry ``model_dims`` (node/edge feature dims, num_actions,
    num_nodes) so the checkpoint can rebuild the model without the dataset —
    this is what :mod:`artificial_system.inspection` reads.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    env_states = {}
    for name, env in (envs or {}).items():
        if not hasattr(env, "state_dict"):
            raise TypeError(f"environment {name!r} does not expose state_dict()")
        env_states[name] = env.state_dict()
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "rng_states": capture_rng_states(),
        "env_states": env_states,
        "global_step": int(global_step),
        "config": config,
        "seed": int(seed),
        "env_info": collect_env_info(),
        "extra": dict(extra or {}),
    }
    torch.save(payload, path)
    return path


def resume_execution_state(
    path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    envs: dict | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Restore a state saved by :func:`save_execution_state` into the given
    live objects; returns the full payload (including ``global_step``)."""
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    restore_rng_states(ckpt.get("rng_states") or {})
    env_states = ckpt.get("env_states") or {}
    for name, env in (envs or {}).items():
        if name in env_states:
            env.load_state_dict(env_states[name])
    return ckpt
