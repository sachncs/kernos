"""Memory-mode accumulation strategies.

Three concrete caches:

* ``Full`` — accumulates the full feature matrix; memory ``O(nm)``.
* ``Stream`` — accumulates ``S = Phi^T Phi`` and ``b = Phi^T y``;
  memory ``O(m^2)``.
* ``Adaptive`` — switches between ``Full`` and ``Stream`` based on a
  user-configurable sample count.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from kernos.core.types import Array, check


class Cache(ABC):
    """Abstract base class for normal-equation accumulators."""

    @abstractmethod
    def accumulate(self, phi: Array, y: Array) -> None: ...

    @abstractmethod
    def equations(self) -> tuple[Array, Array]: ...

    @abstractmethod
    def reset(self) -> None: ...

    @property
    @abstractmethod
    def size(self) -> int: ...


class Full(Cache):
    """Cached feature-matrix accumulator."""

    def __init__(self, dtype: np.dtype = np.float64) -> None:
        self.feats: list[Array] = []
        self.targets: list[Array] = []
        self.dtype = dtype

    def accumulate(self, phi: Array, y: Array) -> None:
        check(phi.ndim == 2, "phi must be 2-D")
        check(y.shape[0] == phi.shape[0], "y and phi must agree on n_samples")
        self.feats.append(phi.astype(self.dtype, copy=False))
        self.targets.append(y.astype(self.dtype, copy=False))

    def equations(self) -> tuple[Array, Array]:
        if not self.feats:
            return np.zeros((0, 0)), np.zeros(0)
        P = np.concatenate(self.feats, axis=0)
        yy = np.concatenate(self.targets, axis=0)
        return P.T @ P, P.T @ yy

    def reset(self) -> None:
        self.feats.clear()
        self.targets.clear()

    @property
    def size(self) -> int:
        return sum(p.shape[0] for p in self.feats)


class Stream(Cache):
    """Streamed ``(S, b)`` accumulator."""

    def __init__(self, mfeat: int, dtype: np.dtype = np.float64) -> None:
        self.dtype = dtype
        self.gramian = np.zeros((mfeat, mfeat), dtype=dtype)
        self.crossvec = np.zeros(mfeat, dtype=dtype)
        self.count = 0

    def accumulate(self, phi: Array, y: Array) -> None:
        check(phi.ndim == 2, "phi must be 2-D")
        check(y.shape[0] == phi.shape[0], "y and phi must agree on n_samples")
        self.gramian += phi.T @ phi
        self.crossvec += phi.T @ y
        self.count += phi.shape[0]

    def equations(self) -> tuple[Array, Array]:
        return self.gramian.copy(), self.crossvec.copy()

    def reset(self) -> None:
        self.gramian[:] = 0
        self.crossvec[:] = 0
        self.count = 0

    @property
    def size(self) -> int:
        return self.count


class Adaptive(Cache):
    """Switch from ``Full`` to ``Stream`` after ``threshold`` samples."""

    def __init__(self, mfeat: int, threshold: int = 10000, dtype: np.dtype = np.float64) -> None:
        self.primary: Cache = Full(dtype=dtype)
        self.fallback: Stream | None = None
        self.mfeat = mfeat
        self.threshold = threshold
        self.dtype = dtype

    def accumulate(self, phi: Array, y: Array) -> None:
        if self.fallback is None and self.primary.size + phi.shape[0] > self.threshold:
            gramian, crossvec = self.primary.equations()
            self.fallback = Stream(self.mfeat, dtype=self.dtype)
            if gramian.shape == (self.mfeat, self.mfeat):
                self.fallback.gramian += gramian
                self.fallback.crossvec += crossvec
            self.fallback.count = self.primary.size
            self.primary = self.fallback
        self.primary.accumulate(phi, y)

    def equations(self) -> tuple[Array, Array]:
        return self.primary.equations()

    def reset(self) -> None:
        if self.fallback is not None:
            self.fallback.reset()
        else:
            self.primary.reset()

    @property
    def size(self) -> int:
        return self.primary.size
