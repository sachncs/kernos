"""Iterative PCG ridge solver."""

from __future__ import annotations

import sys

import numpy as np

from kernos.core.types import Array
from kernos.numeric import pcg


class Iterative:
    """PCG ridge solver with Jacobi preconditioning.

    Emits a warning to stderr when PCG fails to converge within
    ``max_iter``; callers can also inspect the returned ``info`` code.
    """

    def __init__(self, ridge: float, max_iter: int = 1000, tol: float = 1e-6, eps: float = 1e-12) -> None:
        self.ridge = ridge
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps

    def precon(self, S: Array) -> Array:
        """Build the textbook Jacobi preconditioner for ``S``."""
        diag = np.maximum(np.diag(S), self.eps)
        return 1.0 / diag

    def solve(self, phi: Array, y: Array) -> Array:
        """Solve the normal equations via PCG."""
        features = phi.shape[1]
        S = phi.T @ phi + self.ridge * np.eye(features)
        rhs = phi.T @ y
        precon = self.precon(S)
        weights, info = pcg(S, rhs, max_iter=self.max_iter, tol=self.tol, precon=precon)
        if info > 0:
            print(f"warning: PCG did not converge in {self.max_iter} iterations", file=sys.stderr)
        return weights
