"""Sparse k-NN RBF features.

Single ``Rbf`` class.  Vectorized with ``np.put_along_axis`` for O(n*k)
builds rather than Python loops.
"""

from __future__ import annotations

import numpy as np

from kernos.core.types import Array, check


class Rbf:
    """k-NN RBF local corrective feature builder."""

    def __init__(self, ltau: float, lk: int, eps: float = 1e-8) -> None:
        self.ltau = ltau
        self.lk = lk
        self.eps = eps

    def forward(self, U: Array, A: Array) -> Array:
        """Build sparse RBF features.

        Args:
            U: Embeddings ``(samples, dim)``.
            A: Anchors ``(abasis, dim)``.

        Returns:
            Sparse feature matrix of shape ``(samples, abasis)`` with
            ``lk`` non-zero entries per row (one per nearest anchor).
        """
        check(U.ndim == 2, f"U must be 2-D, got shape {U.shape}")
        check(A.ndim == 2, f"A must be 2-D, got shape {A.shape}")
        samples = U.shape[0]
        anchors = A.shape[0]
        neighbors = min(self.lk, anchors)
        diff = U[:, None, :] - A[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1) + self.eps)
        nn_indices = np.argpartition(dist, neighbors - 1, axis=1)[:, :neighbors]
        sparse = np.zeros((samples, anchors), dtype=np.float64)
        np.put_along_axis(
            sparse, nn_indices, np.exp(-self.ltau * np.take_along_axis(dist, nn_indices, axis=1) ** 2), axis=1
        )
        return sparse

    def norm(self, sparse: Array) -> Array:
        """Per-anchor normalization denominators."""
        return np.sum(sparse * sparse, axis=0) + self.eps
