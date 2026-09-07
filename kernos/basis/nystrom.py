"""Nyström basis."""

from __future__ import annotations

import numpy as np

from kernos.core.types import Array, check
from kernos.basis.whitening import Whitening
from kernos.sample import kmeanspp


class Nystrom:
    """Nyström low-rank basis.

    Selects ``mbasis`` landmarks via ``kmeanspp``, builds the kernel-on-
    landmarks matrix, and stores the whitening map.  ``forward(u)``
    returns ``phi_g(u) = k(u, Z) @ Mg``.
    """

    def __init__(self, landmarks: Array, whitening: Array, gamma: float = 1.0) -> None:
        self.landmarks = landmarks
        self.whitening = whitening
        self.gamma = gamma

    @classmethod
    def fromdata(cls, U: Array, mbasis: int, whitening: Whitening, rng: np.random.Generator, gamma: float = 1.0) -> "Nystrom":
        """Build a ``Nystrom`` basis from data via k-means++ landmarks."""
        indices = kmeanspp(U, mbasis, rng)
        Z = U[indices].copy()
        W = whitening.kernel(Z, gamma)
        _, _, Mg, _ = whitening.build(W)
        return cls(Z, Mg, gamma)

    @classmethod
    def fromlandmarks(cls, Z: Array, whitening: Whitening, gamma: float = 1.0) -> "Nystrom":
        """Build a ``Nystrom`` basis from a user-supplied landmark set."""
        W = whitening.kernel(Z, gamma)
        _, _, Mg, _ = whitening.build(W)
        return cls(Z, Mg, gamma)

    def forward(self, u: Array) -> Array:
        """Build global features for query points ``u``.

        Args:
            u: Shape ``(samples, dim)``.

        Returns:
            Shape ``(samples, rank)``.
        """
        check(u.ndim == 2, f"u must be 2-D, got shape {u.shape}")
        diff = u[:, None, :] - self.landmarks[None, :, :]
        squared = np.sum(diff * diff, axis=-1)
        kvec = np.exp(-self.gamma * squared)
        return kvec @ self.whitening
