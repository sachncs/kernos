"""Identity embedder: passthrough.

Used as an ablation control: confirms that the model's gains come from
the basis / refresh machinery, not from the embedder.
"""

from __future__ import annotations

from kernos.core.types import Array


class Identity:
    """Identity embedder: ``forward(x) == x``."""

    def forward(self, x: Array) -> Array:
        """Return ``x`` unchanged."""
        return x

    def params(self) -> tuple[Array, Array]:
        """No-op params for serialization."""
        import numpy as np

        return np.zeros(0), np.zeros(0)

    def setparams(self, theta: Array, bias: Array) -> None:
        """No-op for tests."""
        return None
