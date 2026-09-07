"""Direct Cholesky ridge solver with jitter fallback."""

from __future__ import annotations

import numpy as np

from kernos.core.error import IllConditionedError
from kernos.core.types import Array
from kernos.numeric import chol


class Direct:
    """Direct ridge solver using Cholesky with jitter fallback."""

    def __init__(self, ridge: float, jitter: float, retry: int, factor: float, cap: float, kappa: float) -> None:
        self.ridge = ridge
        self.jitter = jitter
        self.retry = retry
        self.factor = factor
        self.cap = cap
        self.kappa = kappa

    def solve(self, phi: Array, y: Array) -> Array:
        """Solve ``(Phi^T Phi + ridge I) w = Phi^T y`` directly.

        Raises:
            IllConditionedError: When Cholesky fails after ``retry`` attempts.
        """
        features = phi.shape[1]
        S = phi.T @ phi + self.ridge * np.eye(features)
        factor = chol(S, jitter=self.jitter, retry=self.retry, factor=self.factor, cap=self.cap)
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
        rank = phig.shape[1]
        S = phig.T @ phig + self.ridge * np.eye(rank)
        factor = chol(S, jitter=self.jitter, retry=self.retry, factor=self.factor, cap=self.cap)
        rhs = phig.T @ y
        temp = np.linalg.solve(factor, rhs)
        weights = np.linalg.solve(factor.T, temp)
        return y - phig @ weights
