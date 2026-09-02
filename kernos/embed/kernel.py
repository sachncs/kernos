"""Kernel embedder: precomputed-kernel feature map (e.g. RBF) for ablation."""

from __future__ import annotations

import numpy as np

from kernos.core.types import Array


class Kernel:
    """RBF kernel embedder.

    Computes ``forward(x) = exp(-gamma * ||x - centers||^2)`` against a
    fixed set of centers.  Useful for ablation: comparing against
    learned embeddings isolates the contribution of the embedder.
    """

    def __init__(self, centers: Array, gamma: float = 1.0) -> None:
        self.centers: Array = centers
        self.gamma: float = gamma

    def forward(self, x: Array) -> Array:
        """Embed ``x`` against ``self.centers``.

        Args:
            x: Shape ``(samples, input_dim)``.

        Returns:
            Shape ``(samples, n_centers)``.
        """
        diff = x[:, None, :] - self.centers[None, :, :]
        squared = np.sum(diff * diff, axis=-1)
        return np.exp(-self.gamma * squared)
