"""Nyström basis."""

from __future__ import annotations

import numpy as np

from kernos.basis.whitening import Whitening
from kernos.core.types import Array, check
from kernos.sample import kmeanspp


class Nystrom:
    """Nyström low-rank basis.

    Selects ``mbasis`` landmarks via ``kmeanspp``, builds the kernel-on-
    landmarks matrix, and stores the whitening map.  ``forward(u)``
    returns ``phi_g(u) = k(u, Z) @ Mg``.
    """

    def __init__(self, landmarks: Array, whitening: Array, gamma: float = 1.0) -> None:
        """Initialize a Nyström basis from landmarks and whitening weights."""
        check(landmarks.ndim == 2, f"landmarks must be 2-D, got shape {landmarks.shape}")
        check(landmarks.shape[0] > 0 and landmarks.shape[1] > 0, "landmarks must be non-empty")
        check(bool(np.isfinite(landmarks).all()), "landmarks must contain only finite values")
        check(whitening.ndim == 2, f"whitening must be 2-D, got shape {whitening.shape}")
        check(whitening.shape[0] == landmarks.shape[0], "whitening rows must match landmarks")
        check(whitening.shape[1] > 0, "whitening must contain at least one feature")
        check(bool(np.isfinite(whitening).all()), "whitening must contain only finite values")
        check(bool(np.isfinite(gamma)) and gamma > 0.0, "gamma must be finite and positive")
        self.landmarks = landmarks
        self.whitening = whitening
        self.gamma = gamma

    @classmethod
    def fromdata(
        cls, U: Array, mbasis: int, whitening: Whitening, rng: np.random.Generator, gamma: float = 1.0
    ) -> Nystrom:
        """Build a ``Nystrom`` basis from data via k-means++ landmarks."""
        indices = kmeanspp(U, mbasis, rng)
        Z = U[indices].copy()
        W = whitening.kernel(Z, gamma)
        _, _, Mg, _ = whitening.build(W)
        return cls(Z, Mg, gamma)

    @classmethod
    def fromlandmarks(cls, Z: Array, whitening: Whitening, gamma: float = 1.0) -> Nystrom:
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
        check(u.shape[1] == self.landmarks.shape[1], "u feature width must match landmarks")
        check(bool(np.isfinite(u).all()), "u must contain only finite values")
        diff = u[:, None, :] - self.landmarks[None, :, :]
        squared = np.sum(diff * diff, axis=-1)
        kvec = np.exp(-self.gamma * squared)
        return kvec @ self.whitening
