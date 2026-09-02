"""Core module: foundational types, plans, state, errors, RNG."""

from kernos.core.error import AwarenessError, BudgetExceeded, IllConditionedError, ShapeError
from kernos.core.plan import Buffer, Plan
from kernos.core.rng import Rng
from kernos.core.state import Bundle, Continuous, Discrete
from kernos.core.types import Array, asfloat64, check

__all__ = [
    "Array",
    "AwarenessError",
    "BudgetExceeded",
    "Buffer",
    "Bundle",
    "Continuous",
    "Discrete",
    "IllConditionedError",
    "Plan",
    "Rng",
    "ShapeError",
    "asfloat64",
    "check",
]
