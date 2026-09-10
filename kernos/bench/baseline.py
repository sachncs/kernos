"""Baseline models for comparison."""

from __future__ import annotations

import numpy as np

from kernos.basis.nystrom import Nystrom as NystromBasis
from kernos.basis.random import Random as RandomBasis
from kernos.basis.whitening import Whitening
from kernos.core.types import Array
from kernos.predict.predict import Predict
from kernos.solver.direct import Direct


class Baseline:
    """Abstract baseline.  Subclasses must implement ``fit`` and ``predict``."""

    def fit(self, X: Array, y: Array) -> Baseline:
        raise NotImplementedError

    def predict(self, X: Array) -> Array:
        raise NotImplementedError


class Ridge(Baseline):
    """Plain ridge regression on raw inputs."""

    def __init__(self, ridge: float = 1e-3) -> None:
        self.ridge = ridge
        self.predictor: Predict | None = None

    def fit(self, X: Array, y: Array) -> Ridge:
        solver = Direct(self.ridge, 1e-10, 5, 10.0, 1.0, 1e12)
        weights = solver.solve(X, y)
        self.predictor = Predict(weights=weights)
        return self

    def predict(self, X: Array) -> Array:
        return self.predictor.forward(X)


class Nystrom(Baseline):
    """Nyström ridge regression."""

    def __init__(
        self, mbasis: int = 512, ridge: float = 1e-3, seed: int | None = None, gamma: float = 1.0
    ) -> None:
        self.mbasis = mbasis
        self.ridge = ridge
        self.seed = seed
        self.gamma = gamma
        self.predictor: Predict | None = None
        self.basis: NystromBasis | None = None

    def fit(self, X: Array, y: Array) -> Nystrom:
        rng = np.random.default_rng(self.seed)
        whitening = Whitening(1e-6, 1e-5, 1e-8)
        basis = NystromBasis.fromdata(X, self.mbasis, whitening, rng, gamma=self.gamma)
        phig = basis.forward(X)
        solver = Direct(self.ridge, 1e-10, 5, 10.0, 1.0, 1e12)
        weights = solver.solve(phig, y)
        self.basis = basis
        self.predictor = Predict(weights=weights)
        return self

    def predict(self, X: Array) -> Array:
        return self.predictor.forward(self.basis.forward(X))


class Random(Baseline):
    """Random Fourier feature ridge regression."""

    def __init__(
        self, mfeat: int = 1000, gamma: float = 1.0, ridge: float = 1e-3, seed: int | None = None
    ) -> None:
        self.mfeat = mfeat
        self.gamma = gamma
        self.ridge = ridge
        self.seed = seed
        self.predictor: Predict | None = None
        self.basis: RandomBasis | None = None

    def fit(self, X: Array, y: Array) -> Random:
        rng = np.random.default_rng(self.seed)
        basis = RandomBasis.fromdata(X.shape[1], self.mfeat, rng, gamma=self.gamma)
        phig = basis.forward(X)
        solver = Direct(self.ridge, 1e-10, 5, 10.0, 1.0, 1e12)
        weights = solver.solve(phig, y)
        self.basis = basis
        self.predictor = Predict(weights=weights)
        return self

    def predict(self, X: Array) -> Array:
        return self.predictor.forward(self.basis.forward(X))
