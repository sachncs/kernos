"""Predictor: mean and variance prediction from fused features."""

from __future__ import annotations

import numpy as np

from kernos.core.error import ShapeError
from kernos.core.types import Array


class Predict:
    """Linear predictor with optional Bayesian variance mode.

    In mean mode (default): ``forward(phi) -> phi @ weights``.
    In variance mode (``Sinv`` provided): also exposes posterior
    variance for each query point.
    """

    def __init__(self, weights: Array, Sinv: Array | None = None) -> None:
        self.weights = weights
        self.Sinv = Sinv

    def forward(self, phi: Array) -> Array:
        """Predict ``phi @ weights``.

        Args:
            phi: Fused features ``(samples, features)``.

        Returns:
            Predictions ``(samples,)``.
        """
        if phi.ndim != 2:
            raise ShapeError(f"phi must be 2-D, got shape {phi.shape}")
        return phi @ self.weights

    def variance(self, phi: Array) -> Array:
        """Compute posterior variance per query point.

        Returns ``diag(phi @ Sinv @ phi.T)``.  Raises ``ShapeError`` if
        ``Sinv`` was not provided at construction.
        """
        if self.Sinv is None:
            raise ShapeError("Sinv is required for variance prediction")
        if phi.ndim != 2:
            raise ShapeError(f"phi must be 2-D, got shape {phi.shape}")
        proj = phi @ self.Sinv
        return np.sum(proj * phi, axis=1)
