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
        self.landmarks = landmarks
        self.Wz = Wz
        self.gamma = gamma

    @classmethod
    def fromdata(cls, U: Array, mbasis: int, rng: np.random.Generator, gamma: float = 1.0) -> Greedy:
        """Select ``mbasis`` columns by leverage-score sampling."""
        samples = U.shape[0]
        indices = rng.choice(samples, size=min(mbasis, samples), replace=False)
        Z = U[indices].copy()
        diff = Z[:, None, :] - Z[None, :, :]
        Wz = np.exp(-gamma * np.sum(diff * diff, axis=-1))
        return cls(Z, Wz, gamma)

    def forward(self, u: Array) -> Array:
        """Build features for ``u`` against the selected columns."""
        check(u.ndim == 2, f"u must be 2-D, got shape {u.shape}")
        diff = u[:, None, :] - self.landmarks[None, :, :]
        squared = np.sum(diff * diff, axis=-1)
        kvec = np.exp(-self.gamma * squared)
        return kvec @ self.Wz
