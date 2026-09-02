"""Tikhonov regularized orthogonalizer (Iterative)."""

from __future__ import annotations

import numpy as np

from kernos.core.types import Array


class Tikhonov:
    """Tikhonov (iterative) orthogonalizer with stronger regularization."""

    def __init__(self, eta: float, n_iter: int = 3) -> None:
        self.eta = eta
        self.n_iter = n_iter

    def project(self, phig: Array) -> Array:
        """Return the thin operator ``M = phig^T @ (phig phig^T + eta I)^{-1}``
        of shape ``(samples, samples)``.  Use ``forward`` for the
        projection applied to ``phi_l`` directly (avoids materializing
        the full ``(samples, samples)`` projection).
        """
        samples, _ = phig.shape
        return phig.T @ np.linalg.inv(phig @ phig.T + self.eta * np.eye(samples))

    def forward(self, phig: Array, phil: Array) -> Array:
        """Return ``phi_l_perp = phi_l - phig @ project(phig) @ phil``."""
        return phil - phig @ self.project(phig) @ phil

    def check(self, phig: Array, philp: Array, tol: float = 1e-6) -> bool:
        ng = np.linalg.norm(phig, "fro")
        nl = np.linalg.norm(philp, "fro")
        if ng == 0.0 or nl == 0.0:
            return True
        return float(np.linalg.norm(phig.T @ philp, "fro") / (ng * nl)) < tol
