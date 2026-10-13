"""Frobenius-norm drift metric."""

from __future__ import annotations

from kernos.core.types import Array


class Frobenius:
    """Relative Frobenius drift between two matrices."""

    def measure(self, R: Array, Rref: Array) -> float:
        """Return relative Frobenius drift between current and reference matrices."""
        from kernos.numeric import drift

        return drift(R, Rref)
