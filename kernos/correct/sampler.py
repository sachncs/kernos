"""Residual-aware anchor selection.

Single ``Sampler`` class.  Combines coverage and residual weights via a
mix parameter ``amix``; falls back to uniform sampling when both
weights collapse to zero.
"""

from __future__ import annotations

import numpy as np

from kernos.core.types import Array


class Sampler:
    """Residual-aware anchor sampler."""

    def __init__(self, amix: float) -> None:
        self.amix = amix

    def cover(self, distances: Array) -> Array:
        """Coverage weights from kernel distances ``(samples, mbasis)``.

        Returns the per-sample minimum squared distance.
        """
        return np.min(distances * distances, axis=1)

    def resid(self, residuals: Array) -> Array:
        """Residual weights from per-sample residuals ``(samples,)``."""
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
        cover_weights = self.cover(distances)
        resid_weights = self.resid(residuals) if not noresid else np.zeros_like(residuals)
        weights = (1.0 - self.amix) * cover_weights + self.amix * resid_weights
        total = float(weights.sum())
        samples = U.shape[0]
        if total <= 0.0:
            return rng.choice(samples, size=abasis, replace=False)
        probs = weights / total
        return rng.choice(samples, size=abasis, replace=False, p=probs)
