"""Feature scaling: trace-based calibration."""

from __future__ import annotations

import numpy as np

from kernos.core.types import Array, check


class Scaler:
    """Trace-based feature normalization."""

    def __init__(self, eps: float = 1e-8) -> None:
        """Initialize the trace-scaling numerical floor."""
        check(bool(np.isfinite(eps)) and eps > 0.0, "eps must be finite and positive")
        self.eps = eps

    def gnorm(self, phig: Array) -> float:
        """Compute ``c_g = sqrt(trace(Phi_g^T Phi_g) / n + eps)``."""
        self._validate_block(phig, "phig")
        samples = phig.shape[0]
        return float(np.sqrt(np.sum(phig * phig) / samples + self.eps))

    def lnorm(self, phil: Array) -> float:
        """Compute ``c_l`` for the local feature block."""
        self._validate_block(phil, "phil")
        samples = phil.shape[0]
        return float(np.sqrt(np.sum(phil * phil) / samples + self.eps))

    def scale(self, phi: Array, c: float) -> Array:
        """Scale ``phi`` by ``1 / c``."""
        check(bool(np.isfinite(c)) and c > 0.0, f"scale must be finite and positive, got {c}")
        return phi / c

    def _validate_block(self, block: Array, name: str) -> None:
        """Validate a feature block before calculating a normalization."""
        check(block.ndim == 2, f"{name} must be 2-D, got shape {block.shape}")
        check(block.shape[0] > 0, f"{name} must contain at least one sample")
        check(bool(np.isfinite(block).all()), f"{name} must contain only finite values")
