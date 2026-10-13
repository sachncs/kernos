"""Jacobi preconditioner.

Returns the textbook Jacobi preconditioner ``M = 1 / max(diag(A), eps)``,
the standard left-preconditioner for ``scipy``-style ``cg`` conventions.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from kernos.core.types import Array, check


class Jacobi:
    """Textbook Jacobi preconditioner."""

    def __init__(self, eps: float = 1e-12) -> None:
        """Initialize the Jacobi diagonal floor."""
        check(bool(np.isfinite(eps)) and eps > 0.0, "eps must be finite and positive")
        self.eps = eps

    def precon(self, S: Array) -> Array:
        """Return ``M = 1 / max(diag(S), eps)``."""
        diag = np.maximum(np.diag(S), self.eps)
        return cast(Array, 1.0 / diag)
