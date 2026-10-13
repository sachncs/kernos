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
        """Initialize a predictor with solved weights and optional covariance."""
        if weights.ndim != 1 or not np.isfinite(weights).all():
            raise ShapeError("weights must be a finite 1-D vector")
        if Sinv is not None:
            if Sinv.ndim != 2 or Sinv.shape != (weights.size, weights.size):
                raise ShapeError("Sinv must be a square matrix matching weights")
            if not np.isfinite(Sinv).all():
                raise ShapeError("Sinv must contain only finite values")
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
        if phi.shape[1] != self.weights.size:
            raise ShapeError(f"phi has {phi.shape[1]} features; expected {self.weights.size}")
        if not np.isfinite(phi).all():
            raise ShapeError("phi must contain only finite values")
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
        if phi.shape[1] != self.weights.size:
            raise ShapeError(f"phi has {phi.shape[1]} features; expected {self.weights.size}")
        if not np.isfinite(phi).all():
            raise ShapeError("phi must contain only finite values")
        proj = phi @ self.Sinv
        return np.sum(proj * phi, axis=1)
