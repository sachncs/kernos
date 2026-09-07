"""Spectral-norm drift metric."""

from __future__ import annotations

import numpy as np

from kernos.core.types import Array


class Spectral:
    """Relative spectral (operator-2) drift."""

    def measure(self, R: Array, Rref: Array) -> float:
        delta = float(np.linalg.norm(R - Rref, 2))
        baseline = float(np.linalg.norm(Rref, 2))
        if baseline == 0.0:
            return 0.0
        return delta / baseline
