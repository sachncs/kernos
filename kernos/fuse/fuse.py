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
        """Initialize feature scaling and gate defaults."""
        check(bool(np.isfinite(cglobal)) and cglobal > 0.0, "cglobal must be finite and positive")
        check(bool(np.isfinite(clocal)) and clocal > 0.0, "clocal must be finite and positive")
        check(bool(np.isfinite(gateval)) and 0.0 <= gateval <= 1.0, "gateval must be in [0, 1]")
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
        check(phig.shape[0] == phil.shape[0], "phig and phil must have the same sample count")
        check(bool(np.isfinite(phig).all()) and bool(np.isfinite(phil).all()), "features must be finite")
        cglobal = cglobal if cglobal is not None else self.cglobal
        clocal = clocal if clocal is not None else self.clocal
        gateval = gateval if gateval is not None else self.gateval
        check(bool(np.isfinite(cglobal)) and cglobal > 0.0, "cglobal must be finite and positive")
        check(bool(np.isfinite(clocal)) and clocal > 0.0, "clocal must be finite and positive")
        check(bool(np.isfinite(gateval)) and 0.0 <= gateval <= 1.0, "gateval must be in [0, 1]")
        return np.concatenate(
            [np.sqrt(gateval) * phig / cglobal, np.sqrt(1.0 - gateval) * phil / clocal], axis=1
        )

    def split(self, phi: Array, rank: int) -> tuple[Array, Array]:
        """Split fused features back into ``(phig, phil)``."""
        check(phi.ndim == 2, "phi must be 2-D")
        check(isinstance(rank, int) and 0 <= rank <= phi.shape[1], "rank must be within phi feature width")
        return phi[:, :rank], phi[:, rank:]

    @property
    def gatelogit(self) -> float:
        """Current gate logit (exposed for serialization and tests)."""
        return self.gate.gatelogit
