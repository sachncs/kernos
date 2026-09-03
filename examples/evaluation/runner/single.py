"""Single-run helpers: fit/predict with timing, memory tracking, metrics."""

from __future__ import annotations

import time
import tracemalloc
from typing import Any

import numpy as np

from examples.evaluation.reporting.result_record import SingleRunResult
from kernos import Kernos
from kernos.bench.metric import allmetrics


def run_single_kernos(
    model: Kernos,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> SingleRunResult:
    """Fit Kernos on ``(X_train, y_train)``, predict on ``X_test``.

    Tracks wall-clock fit/predict time and peak memory via
    :mod:`tracemalloc`. Computes RMSE/MAE/R² plus a condition proxy
    ``κ(ΦᵀΦ + λI)`` and a discrete refresh count.
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - t0

    metrics = allmetrics(y_test, y_pred)
    from examples.evaluation.runner.diagnostics import compute_condition_proxy

    cond = compute_condition_proxy(model, X_test, model.ridge)
    refresh_count = 1 if model.bundle_ and model.bundle_.discrete.tlast > 0 else 0

    return SingleRunResult(
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        r2=metrics["r2"],
        train_time_sec=train_time,
        predict_time_sec=predict_time,
        peak_mem_mb=peak / (1024 * 1024),
        refresh_count=refresh_count,
        condition_proxy=cond,
    )


def run_single_baseline(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> SingleRunResult:
    """Fit a baseline estimator and return metrics + timing.

    Refresh count is ``0`` and condition proxy is ``NaN`` for baselines.
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - t0

    metrics = allmetrics(y_test, y_pred)
    return SingleRunResult(
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        r2=metrics["r2"],
        train_time_sec=train_time,
        predict_time_sec=predict_time,
        peak_mem_mb=peak / (1024 * 1024),
        refresh_count=0,
        condition_proxy=float("nan"),
    )
