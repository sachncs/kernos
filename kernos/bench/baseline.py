"""Baseline models for comparison."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from kernos.basis.nystrom import Nystrom as NystromBasis
from kernos.basis.random import Random as RandomBasis
from kernos.basis.whitening import Whitening
from kernos.core.types import Array
from kernos.predict.predict import Predict
from kernos.solver.direct import Direct


class Baseline(ABC):
    """Abstract baseline.  Subclasses must implement ``fit`` and ``predict``."""

    @abstractmethod
    def fit(self, X: Array, y: Array) -> Baseline:
        """Fit the baseline and return itself."""
        raise NotImplementedError  # pragma: no cover - abstract contract

    @abstractmethod
    def predict(self, X: Array) -> Array:
        """Predict targets for ``X`` using a fitted baseline."""
        raise NotImplementedError  # pragma: no cover - abstract contract


class Ridge(Baseline):
    """Plain ridge regression on raw inputs."""

    def __init__(self, ridge: float = 1e-3) -> None:
        """Initialize a raw-input ridge baseline."""
        self.ridge = ridge
        self.predictor: Predict | None = None

    def fit(self, X: Array, y: Array) -> Ridge:
        """Fit ridge regression on raw input features."""
        solver = Direct(self.ridge, 1e-10, 5, 10.0, 1.0, 1e12)
        weights = solver.solve(X, y)
        self.predictor = Predict(weights=weights)
        return self

    def predict(self, X: Array) -> Array:
        """Predict with the fitted ridge baseline."""
        if self.predictor is None:
            raise RuntimeError("Ridge baseline is not fitted")
        return self.predictor.forward(X)


class Nystrom(Baseline):
    """Nyström ridge regression."""

    def __init__(
        self, mbasis: int = 512, ridge: float = 1e-3, seed: int | None = None, gamma: float = 1.0
    ) -> None:
        """Initialize a Nyström ridge baseline."""
        self.mbasis = mbasis
        self.ridge = ridge
        self.seed = seed
        self.gamma = gamma
        self.predictor: Predict | None = None
        self.basis: NystromBasis | None = None

    def fit(self, X: Array, y: Array) -> Nystrom:
        """Fit a whitened Nyström feature map and ridge model."""
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
        """Predict with the fitted Nyström baseline."""
        if self.predictor is None or self.basis is None:
            raise RuntimeError("Nystrom baseline is not fitted")
        return self.predictor.forward(self.basis.forward(X))


class Random(Baseline):
    """Random Fourier feature ridge regression."""

    def __init__(
        self, mfeat: int = 1000, gamma: float = 1.0, ridge: float = 1e-3, seed: int | None = None
    ) -> None:
        """Initialize a random Fourier feature ridge baseline."""
        self.mfeat = mfeat
        self.gamma = gamma
        self.ridge = ridge
        self.seed = seed
        self.predictor: Predict | None = None
        self.basis: RandomBasis | None = None

    def fit(self, X: Array, y: Array) -> Random:
        """Fit random Fourier features followed by ridge regression."""
        rng = np.random.default_rng(self.seed)
        basis = RandomBasis.fromdata(X.shape[1], self.mfeat, rng, gamma=self.gamma)
        phig = basis.forward(X)
        solver = Direct(self.ridge, 1e-10, 5, 10.0, 1.0, 1e12)
        weights = solver.solve(phig, y)
        self.basis = basis
        self.predictor = Predict(weights=weights)
        return self

    def predict(self, X: Array) -> Array:
        """Predict with the fitted random-feature baseline."""
        if self.predictor is None or self.basis is None:
            raise RuntimeError("Random baseline is not fitted")
        return self.predictor.forward(self.basis.forward(X))
