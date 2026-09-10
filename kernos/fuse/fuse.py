"""Feature fusion: combine scaled global and local features."""

from __future__ import annotations

import numpy as np

from kernos.core.types import Array, check
from kernos.fuse.gate import Gate


class Fuse:
    """Combine scaled global + local features under a logistic gate."""

    def __init__(
        self,
        cglobal: float = 1.0,
        clocal: float = 1.0,
        gateval: float = 0.5,
        gatelogit: float = 0.0,
        nofreeze: bool = False,
    ) -> None:
        self.cglobal = cglobal
        self.clocal = clocal
        self.gateval = gateval
        self.gate = Gate(gatelogit=gatelogit, nofreeze=nofreeze)

    def forward(
        self,
        phig: Array,
        phil: Array,
        cglobal: float | None = None,
        clocal: float | None = None,
        gateval: float | None = None,
    ) -> Array:
        """Build fused features ``[sqrt(rho) * phig/cg, sqrt(1-rho) * phil/cl]``."""
        check(phig.ndim == 2 and phil.ndim == 2, "phig and phil must be 2-D")
        cglobal = cglobal if cglobal is not None else self.cglobal
        clocal = clocal if clocal is not None else self.clocal
        gateval = gateval if gateval is not None else self.gateval
        return np.concatenate(
            [np.sqrt(gateval) * phig / cglobal, np.sqrt(1.0 - gateval) * phil / clocal], axis=1
        )

    def split(self, phi: Array, rank: int) -> tuple[Array, Array]:
        """Split fused features back into ``(phig, phil)``."""
        return phi[:, :rank], phi[:, rank:]

    @property
    def gatelogit(self) -> float:
        """Current gate logit (exposed for serialization and tests)."""
        return self.gate.gatelogit
