"""Direct Cholesky ridge solver with jitter fallback."""

from __future__ import annotations

from typing import cast

import numpy as np

from kernos.core.diagnostics import SolverEvent
from kernos.core.types import Array, check
from kernos.numeric import chol_with_jitter, condition
from kernos.solver.equation import validate_system


class Direct:
    """Direct ridge solver using Cholesky with jitter fallback."""

    def __init__(
        self,
        ridge: float,
        jitter: float,
        retry: int,
        factor: float,
        cap: float,
        kappa: float,
        check_condition: bool = True,
    ) -> None:
        """Initialize Cholesky regularization and fallback controls."""
        check(bool(np.isfinite(ridge)) and ridge > 0.0, "ridge must be finite and positive")
        check(bool(np.isfinite(jitter)) and jitter >= 0.0, "jitter must be finite and non-negative")
        check(isinstance(retry, int) and retry >= 0, "retry must be a non-negative integer")
        check(bool(np.isfinite(factor)) and factor > 1.0, "factor must be finite and greater than one")
        check(bool(np.isfinite(cap)) and cap >= jitter, "cap must be finite and at least jitter")
        check(bool(np.isfinite(kappa)) and kappa > 0.0, "kappa must be finite and positive")
        check(isinstance(check_condition, bool), "check_condition must be a boolean")
        self.ridge = ridge
        self.jitter = jitter
        self.retry = retry
        self.factor = factor
        self.cap = cap
        self.kappa = kappa
        self.check_condition = check_condition
        self.last_info = 0
        self.last_iterations = 0
        self.last_solver_event: SolverEvent | None = None
        self.last_jitter: float | None = None
        self.last_attempts = 0
        self.last_condition: float | None = None

    def solve(self, phi: Array, y: Array) -> Array:
        """Solve ``(Phi^T Phi + ridge I) w = Phi^T y`` directly.

        Raises:
            IllConditionedError: When Cholesky fails after ``retry`` attempts.
        """
        validate_system(phi, y)
        self.last_info = 0
        self.last_iterations = 0
        self.last_solver_event = None
        self.last_jitter = None
        self.last_attempts = 0
        self.last_condition = None
        features = phi.shape[1]
        S = phi.T @ phi + self.ridge * np.eye(features)
        if self.check_condition:
            self.last_condition = condition(S, self.kappa, name="normal-equation matrix")
        factor, used_jitter, attempts = chol_with_jitter(
            S, jitter=self.jitter, retry=self.retry, factor=self.factor, cap=self.cap
        )
        self.last_jitter = used_jitter if attempts else None
        self.last_attempts = attempts
        self.last_solver_event = SolverEvent.CHOLESKY_JITTER if attempts else None
        rhs = phi.T @ y
        temp = np.linalg.solve(factor, rhs)
        return np.linalg.solve(factor.T, temp)

    def residual(self, phig: Array, y: Array) -> Array:
        """Per-sample residuals using the current basis.

        Args:
            phig: Global features ``(samples, rank)``.
            y: Targets ``(samples,)``.

        Returns:
            Residuals ``(samples,)`` after a single ridge solve.
        """
        validate_system(phig, y)
        self.last_info = 0
        self.last_iterations = 0
        self.last_solver_event = None
        self.last_jitter = None
        self.last_attempts = 0
        self.last_condition = None
        rank = phig.shape[1]
        S = phig.T @ phig + self.ridge * np.eye(rank)
        if self.check_condition:
            self.last_condition = condition(S, self.kappa, name="normal-equation matrix")
        factor, used_jitter, attempts = chol_with_jitter(
            S, jitter=self.jitter, retry=self.retry, factor=self.factor, cap=self.cap
        )
        self.last_jitter = used_jitter if attempts else None
        self.last_attempts = attempts
        self.last_solver_event = SolverEvent.CHOLESKY_JITTER if attempts else None
        rhs = phig.T @ y
        temp = np.linalg.solve(factor, rhs)
        weights = np.linalg.solve(factor.T, temp)
        return cast(Array, y - phig @ weights)
