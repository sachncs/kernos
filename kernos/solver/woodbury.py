"""Woodbury-identity solver for the low-rank regime.

Exploits ``(Phi^T Phi + lambda I)^{-1} = (1/lambda) * I - (1/lambda) * Phi^T
(I + (1/lambda) Phi Phi^T)^{-1} Phi`` to invert a ``(features, features)``
system using a ``(samples, samples)`` inversion.  Useful when
``samples < features`` (over-complete feature space).
"""

from __future__ import annotations

from typing import cast

import numpy as np

from kernos.core.diagnostics import SolverEvent
from kernos.core.error import IllConditionedError
from kernos.core.types import Array, check
from kernos.numeric import chol_with_jitter
from kernos.solver.equation import validate_system


class Woodbury:
    """Woodbury ridge solver for low-rank regimes (``samples < features``)."""

    def __init__(self, ridge: float, jitter: float, retry: int, factor: float, cap: float) -> None:
        """Initialize Woodbury regularization and Cholesky fallback controls."""
        check(bool(np.isfinite(ridge)) and ridge > 0.0, "ridge must be finite and positive")
        check(bool(np.isfinite(jitter)) and jitter >= 0.0, "jitter must be finite and non-negative")
        check(isinstance(retry, int) and retry >= 0, "retry must be a non-negative integer")
        check(bool(np.isfinite(factor)) and factor > 1.0, "factor must be finite and greater than one")
        check(bool(np.isfinite(cap)) and cap >= jitter, "cap must be finite and at least jitter")
        self.ridge = ridge
        self.jitter = jitter
        self.retry = retry
        self.factor = factor
        self.cap = cap
        self.last_info = 0
        self.last_iterations = 0
        self.last_solver_event: SolverEvent | None = None
        self.last_jitter: float | None = None
        self.last_attempts = 0

    def solve(self, phi: Array, y: Array) -> Array:
        """Solve via Woodbury identity."""
        validate_system(phi, y)
        self.last_info = 0
        self.last_iterations = 0
        self.last_solver_event = None
        self.last_jitter = None
        self.last_attempts = 0
        samples, features = phi.shape
        if samples >= features:
            raise IllConditionedError(
                f"Woodbury expects samples < features, got samples={samples}, features={features}"
            )
        lam = self.ridge
        A = np.eye(samples) + (1.0 / lam) * (phi @ phi.T)
        factor, used_jitter, attempts = chol_with_jitter(
            A, jitter=self.jitter, retry=self.retry, factor=self.factor, cap=self.cap
        )
        self.last_jitter = used_jitter if attempts else None
        self.last_attempts = attempts
        self.last_solver_event = SolverEvent.CHOLESKY_JITTER if attempts else None
        rhs = (1.0 / lam) * phi.T @ y - (1.0 / lam**2) * phi.T @ np.linalg.solve(
            factor, np.linalg.solve(factor.T, phi @ phi.T @ y)
        )
        return cast(Array, rhs)

    def residual(self, phi: Array, y: Array) -> Array:
        """Return residuals after a Woodbury ridge solve."""
        validate_system(phi, y)
        self.last_info = 0
        self.last_iterations = 0
        self.last_solver_event = None
        self.last_jitter = None
        self.last_attempts = 0
        return cast(Array, y - phi @ self.solve(phi, y))
