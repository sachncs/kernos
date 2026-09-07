"""Synthetic regression datasets."""

from __future__ import annotations

import numpy as np

from kernos.core.types import Array


def linear(rng: np.random.Generator, samples: int, features: int, noise: float = 0.1) -> tuple[Array, Array]:
    """Linear regression with Gaussian design and Gaussian noise."""
    X = rng.standard_normal((samples, features))
    weights = rng.standard_normal(features)
    y = X @ weights + noise * rng.standard_normal(samples)
    return X, y


def poly(rng: np.random.Generator, samples: int, degree: int, noise: float = 0.1) -> tuple[Array, Array]:
    """Polynomial regression on a single feature."""
    x = rng.standard_normal(samples)
    powers = np.arange(degree + 1)
    coeffs = rng.standard_normal(degree + 1)
    y = np.polyval(coeffs[::-1], x) + noise * rng.standard_normal(samples)
    return x.reshape(-1, 1), y


def highdim(rng: np.random.Generator, samples: int, features: int, informative: int, noise: float = 0.1) -> tuple[Array, Array]:
    """High-dim regression with only ``informative`` columns carrying signal."""
    X = rng.standard_normal((samples, features))
    selected = rng.choice(features, size=informative, replace=False)
    weights = rng.standard_normal(informative)
    y = X[:, selected] @ weights + noise * rng.standard_normal(samples)
    return X, y


def hetero(rng: np.random.Generator, samples: int, base: float = 0.1) -> tuple[Array, Array]:
    """Heteroscedastic regression: noise std grows with ``|x|``."""
    X = rng.standard_normal((samples, 1))
    y = X[:, 0] + (base + np.abs(X[:, 0])) * rng.standard_normal(samples)
    return X, y


def split(X: Array, y: Array, size: float, rng: np.random.Generator | None = None) -> tuple[Array, Array, Array, Array]:
    """Random train/test split.

    Args:
        X: Inputs.
        y: Targets.
        size: Test fraction in ``(0, 1)``.
        rng: NumPy generator (required for determinism).

    Returns:
        ``(X_train, X_test, y_train, y_test)``.
    """
    if rng is None:
        rng = np.random.default_rng()
    samples = X.shape[0]
    n_test = int(round(size * samples))
    perm = rng.permutation(samples)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
