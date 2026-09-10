"""Woodbury-identity solver for the low-rank regime.

Exploits ``(Phi^T Phi + lambda I)^{-1} = (1/lambda) * I - (1/lambda) * Phi^T
(I + (1/lambda) Phi Phi^T)^{-1} Phi`` to invert a ``(features, features)``
system using a ``(samples, samples)`` inversion.  Useful when
``samples < features`` (over-complete feature space).
"""

from __future__ import annotations

import numpy as np

from kernos.core.error import IllConditionedError
from kernos.core.types import Array
from kernos.numeric import chol


class Woodbury:
    """Woodbury ridge solver for low-rank regimes (``samples < features``)."""

    def __init__(self, ridge: float, jitter: float, retry: int, factor: float, cap: float) -> None:
        self.ridge = ridge
        self.jitter = jitter
        self.retry = retry
        self.factor = factor
        self.cap = cap

    def solve(self, phi: Array, y: Array) -> Array:
        """Solve via Woodbury identity."""
        samples, features = phi.shape
        if samples >= features:
            raise IllConditionedError(
                f"Woodbury expects samples < features, got samples={samples}, features={features}"
            )
        lam = self.ridge
        A = np.eye(samples) + (1.0 / lam) * (phi @ phi.T)
        factor = chol(A, jitter=self.jitter, retry=self.retry, factor=self.factor, cap=self.cap)
        rhs = (1.0 / lam) * phi.T @ y - (1.0 / lam**2) * phi.T @ np.linalg.solve(
            factor, np.linalg.solve(factor.T, phi @ phi.T @ y)
        )
        return rhs
