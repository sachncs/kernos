"""State containers for kernos.

Three frozen dataclasses compose the full training state:

* ``Continuous`` — embedding parameters and projection matrix ``R``
  (updated every step via gradient descent).
* ``Discrete`` — landmarks, anchors, whitening map, calibration scalars,
  and refresh-controller metadata (rebuilt wholesale on refresh).
* ``Bundle`` — composes continuous and discrete into a single immutable
  snapshot that flows through the training loop.

All three use ``dataclasses.replace`` so behaviour is uniform and Pythonic.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from kernos.core.types import Array


def _owned_readonly(value: Array | None) -> Array | None:
    """Copy an array at a snapshot boundary and make the copy read-only."""
    if value is None:
        return None
    owned = np.array(value, dtype=np.float64, copy=True)
    owned.setflags(write=False)
    return owned


@dataclass(frozen=True)
class Continuous:
    """Continuously updated representation parameters.

    Attributes:
        theta: Concrete ``Embed`` instance (linear by default).
        R: Projection matrix of shape ``(dim, dim)`` initialized to ``I``.
    """

    theta: Any = None
    R: Array | None = None

    def __post_init__(self) -> None:
        """Own and freeze the projection matrix stored in this snapshot."""
        object.__setattr__(self, "R", _owned_readonly(self.R))


@dataclass(frozen=True)
class Discrete:
    """Discrete basis state rebuilt on refresh.

    Attributes:
        landmarks: ``(mbasis, dim)``.
        anchors: ``(abasis, dim)``.
        whitening: ``(mbasis, rank)`` for the Nyström basis.
        basis_wz: ``(mbasis, mbasis)`` for the Greedy basis; ``None`` otherwise.
        cglobal: Global calibration scalar.
        clocal: Local calibration scalar.
        denoms: Local normalization denominators ``(abasis,)``.
        tlast: Step index of the most recent refresh.
        active: Hysteresis flag (``1`` = active, ``0`` = inactive).
        gateval: Fusion gate value in ``(0, 1)``.
        gatelogit: Gate logit scalar (learnable).
    """

    landmarks: Array | None = None
    anchors: Array | None = None
    whitening: Array | None = None
    basis_wz: Array | None = None
    cglobal: float = 1.0
    clocal: float = 1.0
    denoms: Array | None = None
    tlast: int = 0
    active: int = 1
    gateval: float = 0.5
    gatelogit: float = 0.0

    def __post_init__(self) -> None:
        """Own and freeze all array-backed discrete state."""
        for name in ("landmarks", "anchors", "whitening", "basis_wz", "denoms"):
            object.__setattr__(self, name, _owned_readonly(getattr(self, name)))


@dataclass(frozen=True)
class Bundle:
    """Complete training state passed through the loop.

    Attributes:
        continuous: Continuous representation state.
        discrete: Discrete basis state.
        step: Current training step index (0-based).
        weights: Ridge coefficients ``(rank + abasis,)``; ``None`` before first solve.
    """

    continuous: Continuous = field(default_factory=Continuous)
    discrete: Discrete = field(default_factory=Discrete)
    step: int = 0
    weights: Array | None = None

    def __post_init__(self) -> None:
        """Own and freeze the solved coefficient vector."""
        object.__setattr__(self, "weights", _owned_readonly(self.weights))

    def replace(self, **kwargs: Any) -> Bundle:
        """Return a new ``Bundle`` with the given fields overridden."""
        return replace(self, **kwargs)
