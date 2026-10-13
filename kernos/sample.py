"""Sampling utilities."""

from __future__ import annotations

import numpy as np

from kernos.core.types import Array, check


def kmeanspp(X: Array, k: int, rng: np.random.Generator) -> Array:
    """k-means++ landmark selection.

    Args:
        X: Data matrix of shape ``(samples, dim)``.
        k: Number of landmarks to select.
        rng: NumPy generator.

    Returns:
        Indices of selected landmarks, shape ``(k,)``.

    Raises:
        ValueError: When ``k > samples``.
    """
    check(X.ndim == 2, f"X must be 2-D, got shape {X.shape}")
    check(X.shape[0] > 0 and X.shape[1] > 0, "X must be non-empty")
    check(bool(np.isfinite(X).all()), "X must contain only finite values")
    check(
        isinstance(k, (int, np.integer)) and not isinstance(k, (bool, np.bool_)),
        f"k must be an integer, got {k!r}",
    )
    samples = X.shape[0]
    check(k <= samples, f"k ({k}) must be <= samples ({samples})")
    check(k >= 1, f"k must be >= 1, got {k}")
    indices = np.empty(k, dtype=np.int64)
    indices[0] = int(rng.integers(0, samples))
    closest_sq = np.sum((X - X[indices[0]]) ** 2, axis=1)
    for i in range(1, k):
        total = float(closest_sq.sum())
        if total <= 0.0:
            remaining = np.setdiff1d(np.arange(samples), indices[:i], assume_unique=False)
            indices[i] = int(rng.choice(remaining))
            closest_sq = np.sum((X - X[indices[i]]) ** 2, axis=1)
            continue
        probs = closest_sq / total
        picked = int(rng.choice(samples, p=probs))
        indices[i] = picked
        new_sq = np.sum((X - X[picked]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, new_sq)
    return indices
