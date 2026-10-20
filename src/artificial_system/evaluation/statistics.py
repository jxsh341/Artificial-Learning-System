"""Multi-seed statistics (spec §22).

Never report only the best run: aggregate mean / std / median / CI over
per-seed summaries and keep raw per-seed values.
"""
from __future__ import annotations

import math
from numbers import Number

import numpy as np
from scipy import stats


def confidence_interval(values, confidence: float = 0.95) -> tuple[float, float]:
    """Return ``(mean, half_width)`` of the t-based confidence interval."""
    values = np.asarray(list(values), dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, float("nan")
    half = float(
        stats.t.ppf((1 + confidence) / 2, values.size - 1)
        * values.std(ddof=1)
        / math.sqrt(values.size)
    )
    return mean, half


def aggregate_summaries(summaries: list[dict]) -> dict:
    """Aggregate a list of per-seed summary dicts into a statistics dict."""
    keys = set()
    for s in summaries:
        keys.update(s.keys())
    out: dict = {"n_seeds": len(summaries)}
    for key in sorted(keys):
        values = [s[key] for s in summaries if key in s]
        if values and all(isinstance(v, Number) and not isinstance(v, bool) for v in values):
            arr = np.asarray(values, dtype=float)
            mean, half = confidence_interval(arr)
            out[f"{key}_mean"] = mean
            out[f"{key}_std"] = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else float("nan")
            out[f"{key}_median"] = float(np.nanmedian(arr))
            out[f"{key}_ci95"] = half
            out[f"{key}_per_seed"] = values
    return out
