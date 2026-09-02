"""Regression metrics.

Module-level functions — no class wrapper per the minimal-abstraction
rule.
"""

from __future__ import annotations

import numpy as np

from kernos.core.types import Array


def rmse(y: Array, yhat: Array) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y: Array, yhat: Array) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y - yhat)))


def r2(y: Array, yhat: Array) -> float:
    """Coefficient of determination.

    Returns ``nan`` (rather than 1.0) when ``y`` is constant so a
    trivially-zero model is not silently scored as perfect.
    """
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def maxerr(y: Array, yhat: Array) -> float:
    """Maximum absolute error."""
    return float(np.max(np.abs(y - yhat)))


def allmetrics(y: Array, yhat: Array) -> dict[str, float]:
    """Compute all metrics in one pass."""
    return {"rmse": rmse(y, yhat), "mae": mae(y, yhat), "r2": r2(y, yhat), "maxerr": maxerr(y, yhat)}
