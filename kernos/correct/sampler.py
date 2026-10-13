"""Residual-aware anchor selection.

Single ``Sampler`` class.  Combines coverage and residual weights via a
mix parameter ``amix``; falls back to uniform sampling when both
weights collapse to zero.
"""

from __future__ import annotations

import numpy as np

from kernos.core.types import Array, check


class Sampler:
    """Residual-aware anchor sampler."""

    def __init__(self, amix: float) -> None:
        """Initialize the coverage/residual sampling mixture."""
        check(bool(np.isfinite(amix)) and 0.0 <= amix <= 1.0, "amix must be finite and in [0, 1]")
        self.amix = amix

    def cover(self, distances: Array) -> Array:
        """Coverage weights from kernel distances ``(samples, mbasis)``.

        Returns the per-sample minimum squared distance.
        """
        check(distances.ndim == 2 and distances.shape[1] > 0, "distances must be a non-empty 2-D array")
        check(bool(np.isfinite(distances).all()), "distances must contain only finite values")
        return np.min(distances * distances, axis=1)

    def resid(self, residuals: Array) -> Array:
        """Residual weights from per-sample residuals ``(samples,)``."""
        check(residuals.ndim == 1, "residuals must be 1-D")
        check(bool(np.isfinite(residuals).all()), "residuals must contain only finite values")
        return np.maximum(residuals, 0.0) ** 2

    def pick(
        self,
        U: Array,
        distances: Array,
        residuals: Array,
        abasis: int,
        rng: np.random.Generator,
        noresid: bool = False,
    ) -> Array:
        """Select ``abasis`` anchors from ``U``.

        Args:
            U: Embeddings ``(samples, dim)``.
            distances: Distances to landmarks ``(samples, mbasis)``.
            residuals: Per-sample residuals ``(samples,)``.
            abasis: Number of anchors.
            rng: NumPy generator.
            noresid: If ``True``, use coverage-only weights (ablation).

        Returns:
            Indices of selected anchors, shape ``(abasis,)``.
        """
        check(U.ndim == 2 and U.shape[0] > 0, "U must be a non-empty 2-D array")
        check(distances.shape[0] == U.shape[0], "U and distances must have the same sample count")
        check(residuals.shape[0] == U.shape[0], "U and residuals must have the same sample count")
        check(
            isinstance(abasis, int) and 1 <= abasis <= U.shape[0], "abasis must fit within the sample count"
        )
        check(bool(np.isfinite(U).all()), "U must contain only finite values")
        cover_weights = self.cover(distances)
        resid_weights = self.resid(residuals) if not noresid else np.zeros_like(residuals)
        weights = (1.0 - self.amix) * cover_weights + self.amix * resid_weights
        total = float(weights.sum())
        samples = U.shape[0]
        if total <= 0.0:
            return rng.choice(samples, size=abasis, replace=False)
        probs = weights / total
        return rng.choice(samples, size=abasis, replace=False, p=probs)
