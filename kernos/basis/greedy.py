"""Greedy leverage-score basis."""

from __future__ import annotations

import numpy as np

from kernos.core.types import Array, check


class Greedy:
    """Greedy leverage-score column sampling.

    Iteratively selects columns of the design matrix with probability
    proportional to their leverage score.  Produces a low-rank basis
    without computing the full kernel matrix on landmarks.
    """

    def __init__(self, landmarks: Array, Wz: Array, gamma: float = 1.0) -> None:
        """Initialize a greedy basis from landmarks and its kernel matrix."""
        check(landmarks.ndim == 2, f"landmarks must be 2-D, got shape {landmarks.shape}")
        check(landmarks.shape[0] > 0 and landmarks.shape[1] > 0, "landmarks must be non-empty")
        check(bool(np.isfinite(landmarks).all()), "landmarks must contain only finite values")
        check(
            Wz.ndim == 2 and Wz.shape == (landmarks.shape[0], landmarks.shape[0]),
            "Wz must be square and match landmarks",
        )
        check(bool(np.isfinite(Wz).all()), "Wz must contain only finite values")
        check(bool(np.isfinite(gamma)) and gamma > 0.0, "gamma must be finite and positive")
        self.landmarks = landmarks
        self.Wz = Wz
        self.gamma = gamma

    @classmethod
    def fromdata(cls, U: Array, mbasis: int, rng: np.random.Generator, gamma: float = 1.0) -> Greedy:
        """Select ``mbasis`` columns by leverage-score sampling."""
        check(U.ndim == 2, f"U must be 2-D, got shape {U.shape}")
        check(U.shape[0] > 0 and U.shape[1] > 0, "U must be non-empty")
        check(bool(np.isfinite(U).all()), "U must contain only finite values")
        check(
            isinstance(mbasis, (int, np.integer)) and not isinstance(mbasis, (bool, np.bool_)),
            "mbasis must be an integer",
        )
        check(1 <= mbasis <= U.shape[0], f"mbasis must be between 1 and samples ({U.shape[0]})")
        check(bool(np.isfinite(gamma)) and gamma > 0.0, "gamma must be finite and positive")
        samples = U.shape[0]
        indices = rng.choice(samples, size=mbasis, replace=False)
        Z = U[indices].copy()
        diff = Z[:, None, :] - Z[None, :, :]
        Wz = np.exp(-gamma * np.sum(diff * diff, axis=-1))
        return cls(Z, Wz, gamma)

    def forward(self, u: Array) -> Array:
        """Build features for ``u`` against the selected columns."""
        check(u.ndim == 2, f"u must be 2-D, got shape {u.shape}")
        check(u.shape[1] == self.landmarks.shape[1], "u feature width must match landmarks")
        check(bool(np.isfinite(u).all()), "u must contain only finite values")
        diff = u[:, None, :] - self.landmarks[None, :, :]
        squared = np.sum(diff * diff, axis=-1)
        kvec = np.exp(-self.gamma * squared)
        return kvec @ self.Wz
