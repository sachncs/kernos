"""Whitening map construction.

Single ``Whitening`` class with two methods:
* ``kernel(Z)`` — kernel matrix on landmarks.
* ``build(W)`` — eigendecomposition + soft truncation.
"""

from __future__ import annotations

import numpy as np

from kernos.core.error import ShapeError
from kernos.core.types import Array, check
from kernos.numeric import clipeig, epsilon, retain, truncate


class Whitening:
    """Build a soft-truncated whitening map from a landmark kernel matrix."""

    def __init__(self, tau: float, alpha: float, eps: float) -> None:
        """Initialize spectral truncation and numerical stability thresholds."""
        check(bool(np.isfinite(tau)) and tau > 0.0, "tau must be finite and positive")
        check(bool(np.isfinite(alpha)) and alpha > 0.0, "alpha must be finite and positive")
        check(bool(np.isfinite(eps)) and eps > 0.0, "eps must be finite and positive")
        self.tau = tau
        self.alpha = alpha
        self.eps = eps

    def kernel(self, Z: Array, gamma: float = 1.0) -> Array:
        """RBF kernel matrix on landmarks ``Z``.

        ``W[i, j] = exp(-gamma * ||Z[i] - Z[j]||^2)``.
        """
        check(Z.ndim == 2, f"Z must be 2-D, got shape {Z.shape}")
        check(Z.shape[0] > 0, "Z must contain at least one landmark")
        check(bool(np.isfinite(Z).all()), "Z must contain only finite values")
        check(bool(np.isfinite(gamma)) and gamma > 0.0, "gamma must be finite and positive")
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
        check(W.ndim == 2 and W.shape[0] > 0 and W.shape[0] == W.shape[1], "W must be non-empty and square")
        check(bool(np.isfinite(W).all()), "W must contain only finite values")
        check(np.allclose(W, W.T), "W must be symmetric")
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
