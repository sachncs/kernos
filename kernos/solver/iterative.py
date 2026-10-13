"""Iterative PCG ridge solver."""

from __future__ import annotations

import logging
from typing import cast

import numpy as np

from kernos.core.diagnostics import SolverEvent
from kernos.core.error import IllConditionedError
from kernos.core.types import Array, check
from kernos.numeric import pcg_detailed
from kernos.solver.equation import validate_system

logger = logging.getLogger("kernos.solver")


class Iterative:
    """PCG ridge solver with Jacobi preconditioning.

    Emits a warning to stderr when PCG fails to converge within
    ``max_iter``; callers can also inspect the returned ``info`` code.
    """

    def __init__(self, ridge: float, max_iter: int = 1000, tol: float = 1e-6, eps: float = 1e-12) -> None:
        """Initialize PCG convergence and preconditioner settings."""
        check(bool(np.isfinite(ridge)) and ridge > 0.0, "ridge must be finite and positive")
        check(isinstance(max_iter, int) and max_iter >= 1, "max_iter must be a positive integer")
        check(bool(np.isfinite(tol)) and tol > 0.0, "tol must be finite and positive")
        check(bool(np.isfinite(eps)) and eps > 0.0, "eps must be finite and positive")
        self.ridge = ridge
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps
        self.last_info = 0
        self.last_iterations = 0
        self.last_solver_event: SolverEvent | None = None
        self.last_jitter: float | None = None
        self.last_attempts = 0

    def precon(self, S: Array) -> Array:
        """Build the textbook Jacobi preconditioner for ``S``."""
        diag = np.maximum(np.diag(S), self.eps)
        return cast(Array, 1.0 / diag)

    def solve(self, phi: Array, y: Array) -> Array:
        """Solve the normal equations via PCG."""
        validate_system(phi, y)
        self.last_jitter = None
        self.last_attempts = 0
        features = phi.shape[1]
        S = phi.T @ phi + self.ridge * np.eye(features)
        rhs = phi.T @ y
        precon = self.precon(S)
        weights, info, iterations = pcg_detailed(S, rhs, max_iter=self.max_iter, tol=self.tol, precon=precon)
        self.last_info = int(info)
        self.last_iterations = iterations
        self.last_solver_event = (
            None if info == 0 else SolverEvent.PCG_NONCONVERGED if info > 0 else SolverEvent.PCG_BREAKDOWN
        )
        if info < 0:
            raise IllConditionedError(f"PCG broke down at iteration {abs(int(info))}")
        if info > 0:
            logger.warning("PCG did not converge in %d iterations", self.max_iter)
        return weights

    def residual(self, phi: Array, y: Array) -> Array:
        """Per-sample residuals after a single PCG solve.

        Reuses :meth:`solve` so the residual stays consistent with the
        iterative coefficients (no separate direct solve).
        """
        validate_system(phi, y)
        self.last_jitter = None
        self.last_attempts = 0
        return cast(Array, y - phi @ self.solve(phi, y))
