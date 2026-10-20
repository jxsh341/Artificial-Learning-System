"""Checkpointing with full reproducibility metadata (spec §24, §25)."""
from __future__ import annotations

import platform
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
