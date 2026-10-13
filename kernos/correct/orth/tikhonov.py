"""Tikhonov regularized orthogonalizer (Iterative)."""

from __future__ import annotations

from typing import cast

import numpy as np

from kernos.core.types import Array, check


class Tikhonov:
    """Tikhonov (iterative) orthogonalizer with stronger regularization."""

    def __init__(self, eta: float, n_iter: int = 3) -> None:
        """Initialize Tikhonov regularization and iteration settings."""
        check(bool(np.isfinite(eta)) and eta > 0.0, f"eta must be finite and positive, got {eta}")
        check(isinstance(n_iter, int) and n_iter >= 1, f"n_iter must be a positive integer, got {n_iter}")
        self.eta = eta
        self.n_iter = n_iter

    def _validate(self, phig: Array, phil: Array | None = None) -> None:
        """Validate feature blocks before forming the regularized system."""
        check(phig.ndim == 2, f"phig must be 2-D, got shape {phig.shape}")
        check(phig.shape[0] > 0, "phig must contain at least one sample")
        check(bool(np.isfinite(phig).all()), "phig must contain only finite values")
        if phil is not None:
            check(phil.ndim == 2, f"phil must be 2-D, got shape {phil.shape}")
            check(phil.shape[0] == phig.shape[0], "phig and phil must have the same sample count")
            check(bool(np.isfinite(phil).all()), "phil must contain only finite values")

    def project(self, phig: Array) -> Array:
        """Return the Tikhonov projection operator.

        The operator is ``M = phig^T @ (phig phig^T + eta I)^{-1}`` of shape
        ``(samples, samples)``. Use ``forward`` for the projection applied to
        ``phi_l`` directly to avoid materializing the full projection.
        """
        self._validate(phig)
        samples, _ = phig.shape
        gram = phig @ phig.T + self.eta * np.eye(samples)
        return cast(Array, np.linalg.solve(gram.T, phig).T)

    def forward(self, phig: Array, phil: Array) -> Array:
        """Return ``phi_l_perp = phi_l - phig @ project(phig) @ phil``."""
        self._validate(phig, phil)
        return cast(Array, phil - phig @ self.project(phig) @ phil)

    def check(self, phig: Array, philp: Array, tol: float = 1e-6) -> bool:
        """Return whether the feature blocks meet the orthogonality tolerance."""
        ng = np.linalg.norm(phig, "fro")
        nl = np.linalg.norm(philp, "fro")
        if ng == 0.0 or nl == 0.0:
            return True
        return float(np.linalg.norm(phig.T @ philp, "fro") / (ng * nl)) < tol
