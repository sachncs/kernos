"""Logistic gate for global/local feature fusion.

Single-word API: ``Gate.value(gatelogit)`` returns the gate ``rho`` and
``Gate.sigmoid(a)`` is the underlying logistic.
"""

from __future__ import annotations

from scipy.special import expit


class Gate:
    """Logistic sigmoid gate with a learnable logit."""

    def __init__(self, gatelogit: float = 0.0, nofreeze: bool = False) -> None:
        self.gatelogit = gatelogit
        self.nofreeze = nofreeze

    def sigmoid(self, a: float) -> float:
        """Numerically stable logistic (via ``scipy.special.expit``)."""
        return float(expit(a))

    def value(self, gatelogit: float | None = None) -> float:
        """Return the gate value; updates ``self.gatelogit`` unless ``nofreeze``."""
        if gatelogit is None:
            gatelogit = self.gatelogit
        elif not self.nofreeze:
            self.gatelogit = gatelogit
        return self.sigmoid(gatelogit)
