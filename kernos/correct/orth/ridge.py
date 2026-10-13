"""Ridge-projection orthogonalizer.

Projects ``phi_l`` into the nullspace of ``phi_g`` using a ridge-
regularized pseudo-inverse.  The projection matrix is materialized as
``(samples, rank)`` (not ``(samples, samples)``) for memory efficiency.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from kernos.core.types import Array, check


class Ridge:
    """Ridge-regularized orthogonalizer."""

    def __init__(self, eta: float) -> None:
        """Initialize the ridge orthogonalizer regularization."""
        check(bool(np.isfinite(eta)) and eta > 0.0, f"eta must be finite and positive, got {eta}")
        self.eta = eta

    def _validate(self, phig: Array, phil: Array | None = None) -> None:
        """Validate feature blocks before forming a Gram matrix."""
        check(phig.ndim == 2, f"phig must be 2-D, got shape {phig.shape}")
        check(phig.shape[0] > 0, "phig must contain at least one sample")
        check(bool(np.isfinite(phig).all()), "phig must contain only finite values")
        if phil is not None:
            check(phil.ndim == 2, f"phil must be 2-D, got shape {phil.shape}")
            check(phil.shape[0] == phig.shape[0], "phig and phil must have the same sample count")
            check(bool(np.isfinite(phil).all()), "phil must contain only finite values")

    def project(self, phig: Array) -> Array:
        """Return the thin ridge projection operator.

        The operator is ``M = (phig^T phig + eta I)^{-1} @ phig^T`` of shape
        ``(rank, samples)`` so that ``phig @ M`` is the projection
        ``(samples, samples)`` without materializing it.
        """
        self._validate(phig)
        _, rank = phig.shape
        Sg = phig.T @ phig + self.eta * np.eye(rank)
        factor = np.linalg.cholesky(Sg)
        temp = np.linalg.solve(factor.T, np.linalg.solve(factor, phig.T))
        return temp

    def forward(self, phig: Array, phil: Array) -> Array:
        """Return ``phi_l_perp = phi_l - phig @ project(phig) @ phil``."""
        self._validate(phig, phil)
        return cast(Array, phil - phig @ self.project(phig) @ phil)

    def check(self, phig: Array, philp: Array, tol: float = 1e-6) -> bool:
        """Return ``True`` iff ``||phig^T philp||_F / (||phig||_F ||philp||_F) < tol``."""
        ng = np.linalg.norm(phig, "fro")
        nl = np.linalg.norm(philp, "fro")
        if ng == 0.0 or nl == 0.0:
            return True
        return float(np.linalg.norm(phig.T @ philp, "fro") / (ng * nl)) < tol
