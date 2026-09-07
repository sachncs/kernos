"""Projector: row-normalize then linear project by ``R``.

``forward(u) = normalize(u) @ R.T`` returns a matrix with the same
shape as ``u`` (rows are samples, columns are the projected embedding
dimensions).
"""

from __future__ import annotations

import numpy as np

from kernos.core.types import Array


def norm(x: Array, delta: float = 1e-8) -> Array:
    """Row-normalize ``x`` (shape ``(samples, dim)``).

    Args:
        x: Input matrix.
        delta: Floor added to the squared norm to avoid division by zero.

    Returns:
        Normalized matrix with unit-norm rows.
    """
    squared = np.sum(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(squared + delta)


class Projector:
    """Linear projection by ``R`` after row normalization."""

    def __init__(self, R: Array, delta: float = 1e-8) -> None:
        self.R: Array = R
        self.delta: float = delta

    def forward(self, u: Array) -> Array:
        """Project ``u`` by ``R`` after row normalization.

        Returns a matrix with shape ``(u.shape[0], R.shape[0])``.
        """
        return norm(u, self.delta) @ self.R.T
