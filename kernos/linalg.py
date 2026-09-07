"""Linear-algebra primitives used by kernos solvers.

Single-word names.  ``fnorm`` and ``drift`` live in ``numeric.py``; this
module contains only the Cholesky/PCG helpers that aren't already
imported there.
"""

from __future__ import annotations

import numpy as np

from kernos.core.types import Array
from kernos.numeric import chol, pcg

__all__ = ["chol", "pcg"]
