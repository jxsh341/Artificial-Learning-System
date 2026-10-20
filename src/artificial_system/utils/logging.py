"""Experiment logging (spec §22, §23).

Writes CSV tables and JSON summaries under the seed run directory, and
optionally mirrors scalars to TensorBoard when available.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s %(name)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


class ExperimentLogger:
    """Per-run logger rooted at ``run_dir``."""

    def __init__(self, run_dir: str | Path, use_tensorboard: bool = False):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, list[dict]] = {}
        self._tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(str(self.run_dir / "tensorboard"))
            except Exception as exc:  # pragma: no cover - optional dependency
                get_logger(__name__).warning(f"TensorBoard unavailable ({exc}); using CSV only")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self._tb is not None:
            self._tb.add_scalar(tag, value, step)

    def add_record(self, table: str, record: dict) -> None:
        self._records.setdefault(table, []).append(record)

    def write_table(self, name: str, rows: list[dict] | None = None) -> Path:
        rows = rows if rows is not None else self._records.pop(name, [])
        path = self.run_dir / f"{name}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def save_json(self, name: str, obj) -> Path:
        path = self.run_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, default=_json_default)
        return path

    def close(self) -> None:
        if self._tb is not None:
            self._tb.flush()
            self._tb.close()
