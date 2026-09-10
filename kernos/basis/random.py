"""Random Fourier features basis."""

from __future__ import annotations

import numpy as np

from kernos.core.types import Array, check


class Random:
    """Random Fourier feature basis approximating an RBF kernel.

    ``forward(x) = sqrt(2/mfeat) * cos(omega @ x + phases)``.

    Implements the same API as :class:`Nystrom` so the public estimator
    can swap them transparently.
    """

    def __init__(self, omega: Array, phases: Array, mfeat: int, gamma: float = 1.0) -> None:
        self.omega = omega
        self.phases = phases
        self.mfeat = mfeat
        self.gamma = gamma

    @classmethod
    def fromdata(cls, input_dim: int, mfeat: int, rng: np.random.Generator, gamma: float = 1.0) -> Random:
        """Sample RFF frequencies and phases.

        ``omega ~ N(0, 2*gamma*I)``, ``phases ~ Uniform(0, 2*pi)``.
        """
        omega = rng.standard_normal((mfeat, input_dim)) * np.sqrt(2.0 * gamma)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=mfeat)
        return cls(omega, phases, mfeat, gamma)

    def forward(self, u: Array) -> Array:
        """Build RFF features for ``u`` of shape ``(samples, input_dim)``."""
        check(u.ndim == 2, f"u must be 2-D, got shape {u.shape}")
        projected = u @ self.omega.T + self.phases[None, :]
        return np.sqrt(2.0 / self.mfeat) * np.cos(projected)
