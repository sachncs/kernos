"""Discrete refresh pipeline.

Single ``Refresh`` class with one ``run`` method that rebuilds the
discrete basis from the current projected embeddings ``U`` and targets
``y`` according to ``plan``.
"""

from __future__ import annotations

import numpy as np

from kernos.basis.nystrom import Nystrom
from kernos.basis.whitening import Whitening
from kernos.core.plan import Plan
from kernos.core.state import Bundle, Discrete
from kernos.correct.rbf import Rbf
from kernos.correct.sampler import Sampler
from kernos.embed.projector import Projector
from kernos.fuse.fuse import Fuse
from kernos.fuse.scaler import Scaler
from kernos.sample import kmeanspp
from kernos.solver.direct import Direct


class Refresh:
    """Discrete refresh pipeline."""

    def __init__(self, plan: Plan, whitening: Whitening, scaler: Scaler, sampler: Sampler, rbf: Rbf, fuse: Fuse, solver: Direct) -> None:
        self.plan = plan
        self.whitening = whitening
        self.scaler = scaler
        self.sampler = sampler
        self.rbf = rbf
        self.fuse = fuse
        self.solver = solver

    def run(self, bundle: Bundle, U: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> Discrete:
        """Rebuild ``Discrete`` from projected embeddings ``U``."""
        samples = U.shape[0]
        basis = Nystrom.fromdata(U, self.plan.mbasis, self.whitening, rng)
        landmarks = basis.landmarks
        whitening = basis.whitening
        anchors = self._anchors(U, basis, y, rng) if self.plan.abasis > 0 else np.zeros((0, U.shape[1]))
        phil = self.rbf.forward(U, anchors) if anchors.size > 0 else np.zeros((samples, 0))
        denoms = self.rbf.norm(phil) if phil.size > 0 else np.zeros(0)
        cglobal = self.scaler.gnorm(basis.forward(U))
        clocal = self.scaler.lnorm(phil) if phil.size > 0 else 1.0
        fused = self.fuse.forward(basis.forward(U), phil, cglobal=cglobal, clocal=clocal, gateval=0.5)
        self.solver.solve(fused, y)
        gateval = self.fuse.gateval
        return Discrete(
            landmarks=landmarks,
            anchors=anchors if anchors.size > 0 else None,
            whitening=whitening,
            cglobal=cglobal,
            clocal=clocal,
            denoms=denoms,
            tlast=bundle.step,
            active=1,
            gateval=gateval,
            gatelogit=self.fuse.gatelogit,
        )

    def _anchors(self, U: np.ndarray, basis: Nystrom, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Select anchors via the residual-aware Sampler."""
        if self.plan.noresid:
            indices = kmeanspp(U, self.plan.abasis, rng)
            return U[indices]
        phig = basis.forward(U)
        residuals = self.solver.residual(phig, y)
        distances = np.sqrt(np.sum((U[:, None, :] - basis.landmarks[None, :, :]) ** 2, axis=-1))
        indices = self.sampler.pick(U, distances, residuals, self.plan.abasis, rng, noresid=False)
        return U[indices]
