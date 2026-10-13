"""Random Fourier features basis."""

from __future__ import annotations

from typing import cast

import numpy as np

from kernos.core.types import Array, check


class Random:
    """Random Fourier feature basis approximating an RBF kernel.

    ``forward(x) = sqrt(2/mfeat) * cos(omega @ x + phases)``.

    Implements the same API as :class:`Nystrom` so the public estimator
    can swap them transparently.
    """

    def __init__(self, omega: Array, phases: Array, mfeat: int, gamma: float = 1.0) -> None:
        """Initialize a random Fourier feature map."""
        check(
            omega.ndim == 2 and omega.shape[0] > 0 and omega.shape[1] > 0,
            "omega must be a non-empty 2-D array",
        )
        check(phases.ndim == 1 and phases.shape[0] == omega.shape[0], "phases must match omega rows")
        check(
            isinstance(mfeat, (int, np.integer)) and not isinstance(mfeat, (bool, np.bool_)),
            "mfeat must be an integer",
        )
        check(mfeat == omega.shape[0] and mfeat > 0, "mfeat must match the positive number of omega rows")
        check(
            bool(np.isfinite(omega).all()) and bool(np.isfinite(phases).all()),
            "omega and phases must contain only finite values",
        )
        check(bool(np.isfinite(gamma)) and gamma > 0.0, "gamma must be finite and positive")
        self.omega = omega
        self.phases = phases
        self.mfeat = mfeat
        self.gamma = gamma

    @classmethod
    def fromdata(cls, input_dim: int, mfeat: int, rng: np.random.Generator, gamma: float = 1.0) -> Random:
        """Sample RFF frequencies and phases.

        ``omega ~ N(0, 2*gamma*I)``, ``phases ~ Uniform(0, 2*pi)``.
        """
        check(
            isinstance(input_dim, (int, np.integer))
            and not isinstance(input_dim, (bool, np.bool_))
            and input_dim > 0,
            "input_dim must be a positive integer",
        )
        check(
            isinstance(mfeat, (int, np.integer)) and not isinstance(mfeat, (bool, np.bool_)) and mfeat > 0,
            "mfeat must be a positive integer",
        )
        check(bool(np.isfinite(gamma)) and gamma > 0.0, "gamma must be finite and positive")
        omega = rng.standard_normal((mfeat, input_dim)) * np.sqrt(2.0 * gamma)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=mfeat)
        return cls(omega, phases, mfeat, gamma)

    def forward(self, u: Array) -> Array:
        """Build RFF features for ``u`` of shape ``(samples, input_dim)``."""
        check(u.ndim == 2, f"u must be 2-D, got shape {u.shape}")
        check(u.shape[1] == self.omega.shape[1], "u feature width must match omega columns")
        check(bool(np.isfinite(u).all()), "u must contain only finite values")
        projected = u @ self.omega.T + self.phases[None, :]
        return cast(Array, np.sqrt(2.0 / self.mfeat) * np.cos(projected))
