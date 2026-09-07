"""Diagnostics: condition-proxy κ(ΦᵀΦ + λI) and ridge λ tuning."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from kernos import Kernos

FitPredictModel = Any
"""Type alias for any estimator with ``fit`` and ``predict`` (used in lambdas)."""


def compute_condition_proxy(estimator: Kernos, X: np.ndarray, ridge: float) -> float:
    """Compute ``cond(Phi^T Phi + ridge I)`` as a numeric diagnostic.

    Returns ``inf`` if the estimator is not fitted or lacks an
    embedding. Returns ``inf`` if the eigenvalue decomposition fails.
    """
    bundle = getattr(estimator, "bundle_", None)
    if bundle is None or bundle.weights is None:
        return float("inf")
    embed = bundle.continuous.theta
    if embed is None:
        return float("inf")
    from kernos.loop.loop import Loop

    loop = Loop(estimator.plan_)
    _, phi, _, _ = loop.features(bundle, X)
    gram = phi.T @ phi + ridge * np.eye(phi.shape[1])
    try:
        eigs = np.linalg.eigvalsh(gram)
        return float(np.max(eigs) / (np.min(eigs) + 1e-15))
    except Exception:
        return float("inf")


def tune_lambda_reg(
    factory: Callable[[float], FitPredictModel],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lambdas: list[float] | None = None,
) -> float:
    """Pick the ridge ``λ`` that minimises validation RMSE.

    Args:
        factory: Callable taking ``λ`` and returning a freshly
            constructed estimator.
        X_train: Training features.
        y_train: Training targets.
        X_val: Validation features.
        y_val: Validation targets.
        lambdas: Candidate ridge values. Defaults to a small log-spaced
            grid ``[1e-4, …, 1.0]`` when omitted.
    """
    if lambdas is None:
        lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    best_rmse = float("inf")
    best_lam = lambdas[0]
    for lam in lambdas:
        model = factory(lam)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = float(np.sqrt(np.mean((y_val - y_pred) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_lam = lam
    return best_lam
