"""Feature scaling: trace-based calibration."""

from __future__ import annotations

import numpy as np

from kernos.core.types import Array


class Scaler:
    """Trace-based feature normalization."""

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    def gnorm(self, phig: Array) -> float:
        """Compute ``c_g = sqrt(trace(Phi_g^T Phi_g) / n + eps)``."""
        samples = phig.shape[0]
        return float(np.sqrt(np.sum(phig * phig) / samples + self.eps))

    def lnorm(self, phil: Array) -> float:
        """Compute ``c_l`` for the local feature block."""
        samples = phil.shape[0]
        return float(np.sqrt(np.sum(phil * phil) / samples + self.eps))

    def scale(self, phi: Array, c: float) -> Array:
        """Scale ``phi`` by ``1 / c``."""
        return phi / c
