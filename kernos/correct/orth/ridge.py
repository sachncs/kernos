"""Ridge-projection orthogonalizer.

Projects ``phi_l`` into the nullspace of ``phi_g`` using a ridge-
regularized pseudo-inverse.  The projection matrix is materialized as
``(samples, rank)`` (not ``(samples, samples)``) for memory efficiency.
"""

from __future__ import annotations

import numpy as np

from kernos.core.types import Array


class Ridge:
    """Ridge-regularized orthogonalizer."""

    def __init__(self, eta: float) -> None:
        self.eta = eta

    def project(self, phig: Array) -> Array:
        """Return the thin operator ``M = (phig^T phig + eta I)^{-1} @ phig^T``
        of shape ``(rank, samples)`` so that ``phig @ M`` is the
        projection ``(samples, samples)`` without materializing it.
        """
        _, rank = phig.shape
        Sg = phig.T @ phig + self.eta * np.eye(rank)
        factor = np.linalg.cholesky(Sg)
        temp = np.linalg.solve(factor.T, np.linalg.solve(factor, phig.T))
        return temp

    def forward(self, phig: Array, phil: Array) -> Array:
        """Return ``phi_l_perp = phi_l - phig @ project(phig) @ phil``."""
        return phil - phig @ self.project(phig) @ phil

    def check(self, phig: Array, philp: Array, tol: float = 1e-6) -> bool:
        """Return ``True`` iff ``||phig^T philp||_F / (||phig||_F ||philp||_F) < tol``."""
        ng = np.linalg.norm(phig, "fro")
        nl = np.linalg.norm(philp, "fro")
        if ng == 0.0 or nl == 0.0:
            return True
        return float(np.linalg.norm(phig.T @ philp, "fro") / (ng * nl)) < tol
