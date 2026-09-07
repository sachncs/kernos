"""Whitening map construction.

Single ``Whitening`` class with two methods:
* ``kernel(Z)`` — kernel matrix on landmarks.
* ``build(W)`` — eigendecomposition + soft truncation.
"""

from __future__ import annotations

import numpy as np

from kernos.core.error import ShapeError
from kernos.core.types import Array
from kernos.numeric import clipeig, epsilon, retain, truncate


class Whitening:
    """Build a soft-truncated whitening map from a landmark kernel matrix."""

    def __init__(self, tau: float, alpha: float, eps: float) -> None:
        self.tau = tau
        self.alpha = alpha
        self.eps = eps

    def kernel(self, Z: Array, gamma: float = 1.0) -> Array:
        """RBF kernel matrix on landmarks ``Z``.

        ``W[i, j] = exp(-gamma * ||Z[i] - Z[j]||^2)``.
        """
        diff = Z[:, None, :] - Z[None, :, :]
        squared = np.sum(diff * diff, axis=-1)
        return np.exp(-gamma * squared)

    def build(self, W: Array) -> tuple[Array, Array, Array, int]:
        """Eigendecompose ``W`` and return a soft-truncated whitening map.

        Returns:
            (eigenvectors, eigenvalues, whitening, rank) where:
              - ``eigenvectors``: shape ``(landmarks, landmarks)``.
              - ``eigenvalues``: shape ``(landmarks,)``.
              - ``whitening``: shape ``(landmarks, rank)``.
              - ``rank``: retained rank.

        Raises:
            ShapeError: When ``W`` is degenerate (``rank == 0``).
        """
        eigenvalues, eigenvectors = np.linalg.eigh(W)
        eigenvalues = clipeig(eigenvalues, 0.0)
        indices, rank = retain(eigenvalues, self.tau)
        if rank == 0:
            raise ShapeError(f"whitening map degenerate: tau={self.tau}, max eval={eigenvalues.max():.3e}")
        ev_used = truncate(eigenvalues[indices], self.tau, self.eps)
        trace = float(ev_used.sum())
        eps_w = epsilon(trace, indices.size, self.alpha)
        whitening = eigenvectors[:, indices] * np.sqrt(ev_used / (ev_used + eps_w))[None, :]
        return eigenvectors, eigenvalues, whitening, rank
