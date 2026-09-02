"""Linear embedder: ``forward(x) = x @ weights + bias``."""

from __future__ import annotations

import numpy as np

from kernos.core.types import Array


class Linear:
    """Linear dense embedder.

    Stores ``weights`` of shape ``(input_dim, output_dim)`` and ``bias``
    of shape ``(output_dim,)``.  ``forward`` accepts both 1-D and 2-D
    input via a single matrix product.
    """

    def __init__(self, input_dim: int, output_dim: int, rng: np.random.Generator) -> None:
        scale = np.sqrt(2.0 / max(input_dim, 1))
        self.weights: Array = rng.standard_normal((input_dim, output_dim)) * scale
        self.bias: Array = np.zeros(output_dim, dtype=np.float64)

    def forward(self, x: Array) -> Array:
        """Embed ``x`` of shape ``(n, input_dim)`` or ``(input_dim,)``."""
        return x @ self.weights + self.bias

    def params(self) -> tuple[Array, Array]:
        """Return ``(weights, bias)`` for serialization."""
        return self.weights, self.bias

    def setparams(self, weights: Array, bias: Array) -> None:
        """Replace ``weights`` and ``bias`` (used by tests)."""
        self.weights = weights
        self.bias = bias
