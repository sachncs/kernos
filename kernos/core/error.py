"""Exception hierarchy for kernos.

Single-word names: every public exception ends in ``Error`` only when the
module context already implies the kind (e.g. ``IllConditionedError`` lives
in ``kernos.core.error`` so ``Error`` is part of the standard Python
convention).
"""

from __future__ import annotations


class AwarenessError(Exception):
    """Base class for all kernos-raised exceptions."""


class IllConditionedError(AwarenessError):
    """Raised when a numerical system becomes singular or near-singular.

    Surfaces from ``Solver.solve`` (Cholesky failure after jitter retry)
    and from ``Numeric.condition`` (condition number exceeds the plan
    threshold).
    """


class BudgetExceeded(AwarenessError):
    """Raised when a refresh would push the running total above the
    amortized refresh budget configured on ``Plan.budget``."""


class ShapeError(AwarenessError, ValueError):
    """Raised on shape mismatches that cannot be recovered silently.

    Subclass of both ``AwarenessError`` and ``ValueError`` so callers
    catching either will see it.
    """
