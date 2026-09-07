"""Normal equation assembly.

Module-level functions — no class wrapper per the minimal-abstraction
rule.
"""

from __future__ import annotations

import numpy as np

from kernos.core.types import Array


def gramian(phi: Array) -> Array:
    """Return ``Phi^T Phi`` (shape ``(m, m)``)."""
    return phi.T @ phi


def crossvec(phi: Array, y: Array) -> Array:
    """Return ``Phi^T y`` (shape ``(m,)``)."""
    return phi.T @ y


def assemble(phi: Array, y: Array, ridge: float) -> tuple[Array, Array]:
    """Build ridge-stabilized normal equations ``(S, b)``.

    Returns:
        (S, b) where ``S = Phi^T Phi + ridge * I`` and ``b = Phi^T y``.
    """
    m = phi.shape[1]
    S = gramian(phi) + ridge * np.eye(m)
    b = crossvec(phi, y)
    return S, b
