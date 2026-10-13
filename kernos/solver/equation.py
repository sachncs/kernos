"""Normal equation assembly.

Module-level functions — no class wrapper per the minimal-abstraction
rule.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from kernos.core.error import ShapeError
from kernos.core.types import Array


def validate_system(phi: Array, y: Array) -> None:
    """Validate a feature matrix and target vector before numerical work.

    Raises:
        ShapeError: If dimensions, sample counts or finiteness are invalid.
    """
    if phi.ndim != 2:
        raise ShapeError(f"phi must be 2-D (samples, features); got shape {phi.shape}")
    if y.ndim != 1:
        raise ShapeError(f"y must be 1-D (samples,); got shape {y.shape}")
    if phi.shape[0] == 0:
        raise ShapeError("phi and y must contain at least one sample")
    if phi.shape[0] != y.shape[0]:
        raise ShapeError(f"phi and y disagree on samples: {phi.shape[0]} != {y.shape[0]}")
    if not np.isfinite(phi).all() or not np.isfinite(y).all():
        raise ShapeError("phi and y must contain only finite values")


def gramian(phi: Array) -> Array:
    """Return ``Phi^T Phi`` (shape ``(m, m)``)."""
    return cast(Array, phi.T @ phi)


def crossvec(phi: Array, y: Array) -> Array:
    """Return ``Phi^T y`` (shape ``(m,)``)."""
    return cast(Array, phi.T @ y)


def assemble(phi: Array, y: Array, ridge: float) -> tuple[Array, Array]:
    """Build ridge-stabilized normal equations ``(S, b)``.

    Returns:
        (S, b) where ``S = Phi^T Phi + ridge * I`` and ``b = Phi^T y``.
    """
    validate_system(phi, y)
    m = phi.shape[1]
    S = gramian(phi) + ridge * np.eye(m)
    b = crossvec(phi, y)
    return S, b
