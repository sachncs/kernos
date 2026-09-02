"""Jacobi preconditioner.

Returns the textbook Jacobi preconditioner ``M = 1 / max(diag(A), eps)``,
the standard left-preconditioner for ``scipy``-style ``cg`` conventions.
"""

from __future__ import annotations

import numpy as np

from kernos.core.types import Array


class Jacobi:
    """Textbook Jacobi preconditioner."""

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    def precon(self, S: Array) -> Array:
        """Return ``M = 1 / max(diag(S), eps)``."""
        diag = np.maximum(np.diag(S), self.eps)
        return 1.0 / diag
