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

from kernos.core.types import Array


@dataclass(frozen=True)
class Continuous:
    """Continuously updated representation parameters.

    Attributes:
        theta: Concrete ``Embed`` instance (linear by default).
        R: Projection matrix of shape ``(dim, dim)`` initialized to ``I``.
    """

    theta: Any = None
    R: Array | None = None


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

    def replace(self, **kwargs: Any) -> "Bundle":
        """Return a new ``Bundle`` with the given fields overridden."""
        return replace(self, **kwargs)
