"""Projector: row-normalize then linear project by ``R``.

``forward(u) = normalize(u) @ R.T`` returns a matrix with the same
shape as ``u`` (rows are samples, columns are the projected embedding
dimensions).
"""

from __future__ import annotations

from typing import cast

import numpy as np

from kernos.core.types import Array, check


def norm(x: Array, delta: float = 1e-8) -> Array:
    """Row-normalize ``x`` (shape ``(samples, dim)``).

    Args:
        x: Input matrix.
        delta: Floor added to the squared norm to avoid division by zero.

    Returns:
        Normalized matrix with unit-norm rows.
    """
    check(x.ndim == 2, f"x must be 2-D, got shape {x.shape}")
    check(bool(np.isfinite(x).all()), "x must contain only finite values")
    check(bool(np.isfinite(delta)) and delta > 0.0, f"delta must be finite and positive, got {delta}")
    squared = np.sum(x * x, axis=-1, keepdims=True)
    return cast(Array, x / np.sqrt(squared + delta))


class Projector:
    """Linear projection by ``R`` after row normalization."""

    def __init__(self, R: Array, delta: float = 1e-8) -> None:
        """Initialize a row-normalizing projection matrix."""
        check(R.ndim == 2, f"R must be 2-D, got shape {R.shape}")
        check(bool(np.isfinite(R).all()), "R must contain only finite values")
        check(bool(np.isfinite(delta)) and delta > 0.0, f"delta must be finite and positive, got {delta}")
        self.R: Array = R
        self.delta: float = delta

    def forward(self, u: Array) -> Array:
        """Project ``u`` by ``R`` after row normalization.

        Returns a matrix with shape ``(u.shape[0], R.shape[0])``.
        """
        check(u.ndim == 2, f"u must be 2-D, got shape {u.shape}")
        check(u.shape[1] == self.R.shape[1], "u and R must agree on input dimension")
        return cast(Array, norm(u, self.delta) @ self.R.T)
