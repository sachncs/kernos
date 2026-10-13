"""Discrete refresh pipeline.

Single ``Refresh`` class with one ``run`` method that rebuilds the
discrete basis from the current projected embeddings ``U`` and targets
``y`` according to ``plan``.
"""

from __future__ import annotations

import numpy as np

from kernos.basis.whitening import Whitening
from kernos.core.interfaces import Basis, ResidualSolver
from kernos.core.plan import Plan
from kernos.core.state import Bundle, Discrete
from kernos.correct.rbf import Rbf
from kernos.correct.sampler import Sampler
from kernos.fuse.fuse import Fuse
from kernos.fuse.scaler import Scaler
from kernos.sample import kmeanspp


class Refresh:
    """Discrete refresh pipeline."""

    def __init__(
        self,
        plan: Plan,
        whitening: Whitening,
        scaler: Scaler,
        sampler: Sampler,
        rbf: Rbf,
        fuse: Fuse,
        solver: ResidualSolver,
    ) -> None:
        """Initialize the discrete refresh pipeline dependencies."""
        self._plan = plan
        self._whitening = whitening
        self._scaler = scaler
        self._sampler = sampler
        self._rbf = rbf
        self._fuse = fuse
        self._solver = solver

    def run(self, bundle: Bundle, U: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> Discrete:
        """Rebuild ``Discrete`` from projected embeddings ``U``.

        The basis factory (Nyström vs Greedy) is selected by
        ``plan.basis`` and the accumulation strategy is selected by
        ``plan.mode`` so the cache actually affects how the fused
        features are aggregated before solving.
        """
        # Import lazily to avoid a cycle while the core factory registers
        # policy drift implementations during package initialization.
        from kernos.core.factory import build_basis, build_cache

        samples = U.shape[0]
        basis = build_basis(self._plan.basis, U, self._plan, self._whitening, rng)
        basis_wz = getattr(basis, "Wz", None)
        landmarks = basis.landmarks
        whitening: np.ndarray | None = getattr(basis, "whitening", None)
        anchors = self._anchors(U, basis, y, rng) if self._plan.abasis > 0 else np.zeros((0, U.shape[1]))
        phil = self._rbf.forward(U, anchors) if anchors.size > 0 else np.zeros((samples, 0))
        denoms = self._rbf.norm(phil) if phil.size > 0 else np.zeros(0)
        phig = basis.forward(U)
        cglobal = self._scaler.gnorm(phig)
        clocal = self._scaler.lnorm(phil) if phil.size > 0 else 1.0
        fused = self._fuse.forward(phig, phil, cglobal=cglobal, clocal=clocal, gateval=0.5)
        mfeat = fused.shape[1]
        cache = build_cache(self._plan.mode, mfeat, threshold=2 * mfeat)
        cache.accumulate(fused, y)
        _S, _b = cache.equations()
        self._solver.solve(fused, y)
        gateval = self._fuse.gateval
        return Discrete(
            landmarks=landmarks,
            anchors=anchors if anchors.size > 0 else None,
            whitening=whitening,
            basis_wz=basis_wz,
            cglobal=cglobal,
            clocal=clocal,
            denoms=denoms,
            tlast=bundle.step,
            active=1,
            gateval=gateval,
            gatelogit=self._fuse.gatelogit,
        )

    def _anchors(self, U: np.ndarray, basis: Basis, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Select anchors via the residual-aware Sampler."""
        if self._plan.noresid:
            indices = kmeanspp(U, self._plan.abasis, rng)
            return U[indices]
        phig = basis.forward(U)
        residuals = self._solver.residual(phig, y)
        distances = np.sqrt(np.sum((U[:, None, :] - basis.landmarks[None, :, :]) ** 2, axis=-1))
        indices = self._sampler.pick(U, distances, residuals, self._plan.abasis, rng, noresid=False)
        return U[indices]
