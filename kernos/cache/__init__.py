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
from typing import Any

import numpy as np

from kernos.core.types import Array, check


def _validate_batch(phi: Array, y: Array, expected_features: int | None = None) -> None:
    """Validate one feature/target batch before cache mutation."""
    check(phi.ndim == 2, "phi must be 2-D")
    check(y.ndim == 1, "y must be 1-D")
    check(y.shape[0] == phi.shape[0], "y and phi must agree on n_samples")
    check(phi.shape[0] > 0, "phi must contain at least one sample")
    if expected_features is not None:
        check(phi.shape[1] == expected_features, "phi has an incompatible feature width")
    check(bool(np.isfinite(phi).all()), "phi must contain only finite values")
    check(bool(np.isfinite(y).all()), "y must contain only finite values")


class Cache(ABC):
    """Abstract base class for normal-equation accumulators."""

    @abstractmethod
    def accumulate(self, phi: Array, y: Array) -> None:
        """Accumulate one feature/target batch."""

    @abstractmethod
    def equations(self) -> tuple[Array, Array]:
        """Return the accumulated Gramian and cross-vector."""

    @abstractmethod
    def reset(self) -> None:
        """Clear accumulated data."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Return the number of accumulated samples."""


class Full(Cache):
    """Cached feature-matrix accumulator."""

    def __init__(self, dtype: Any = np.float64) -> None:
        """Initialize an empty full-matrix accumulator."""
        self.feats: list[Array] = []
        self.targets: list[Array] = []
        self.dtype = dtype

    def accumulate(self, phi: Array, y: Array) -> None:
        """Append a feature/target batch to the cache."""
        _validate_batch(phi, y)
        self.feats.append(phi.astype(self.dtype, copy=False))
        self.targets.append(y.astype(self.dtype, copy=False))

    def equations(self) -> tuple[Array, Array]:
        """Build normal-equation components from all cached batches."""
        if not self.feats:
            return np.zeros((0, 0)), np.zeros(0)
        P = np.concatenate(self.feats, axis=0)
        yy = np.concatenate(self.targets, axis=0)
        return P.T @ P, P.T @ yy

    def reset(self) -> None:
        """Discard all cached batches."""
        self.feats.clear()
        self.targets.clear()

    @property
    def size(self) -> int:
        """Return the cached sample count."""
        return sum(p.shape[0] for p in self.feats)


class Stream(Cache):
    """Streamed ``(S, b)`` accumulator."""

    def __init__(self, mfeat: int, dtype: Any = np.float64) -> None:
        """Initialize zeroed streamed normal-equation accumulators."""
        check(isinstance(mfeat, int) and mfeat > 0, f"mfeat must be a positive integer, got {mfeat}")
        self.dtype = dtype
        self.gramian = np.zeros((mfeat, mfeat), dtype=dtype)
        self.crossvec = np.zeros(mfeat, dtype=dtype)
        self.count = 0

    def accumulate(self, phi: Array, y: Array) -> None:
        """Update the Gramian and cross-vector with one batch."""
        _validate_batch(phi, y, expected_features=self.gramian.shape[0])
        self.gramian += phi.T @ phi
        self.crossvec += phi.T @ y
        self.count += phi.shape[0]

    def equations(self) -> tuple[Array, Array]:
        """Return copies of the streamed normal-equation components."""
        return self.gramian.copy(), self.crossvec.copy()

    def reset(self) -> None:
        """Zero the streamed accumulators and sample count."""
        self.gramian[:] = 0
        self.crossvec[:] = 0
        self.count = 0

    @property
    def size(self) -> int:
        """Return the streamed sample count."""
        return self.count


class Adaptive(Cache):
    """Switch from ``Full`` to ``Stream`` after ``threshold`` samples."""

    def __init__(self, mfeat: int, threshold: int = 10000, dtype: Any = np.float64) -> None:
        """Initialize an accumulator that switches at ``threshold`` samples."""
        check(isinstance(mfeat, int) and mfeat > 0, f"mfeat must be a positive integer, got {mfeat}")
        check(
            isinstance(threshold, int) and threshold > 0,
            f"threshold must be a positive integer, got {threshold}",
        )
        self.primary: Cache = Full(dtype=dtype)
        self.fallback: Stream | None = None
        self.mfeat = mfeat
        self.threshold = threshold
        self.dtype = dtype

    def accumulate(self, phi: Array, y: Array) -> None:
        """Accumulate data and switch to streaming mode when needed."""
        _validate_batch(phi, y, expected_features=self.mfeat)
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
        """Return normal equations from the active accumulator."""
        return self.primary.equations()

    def reset(self) -> None:
        """Reset the active accumulator while retaining its mode."""
        if self.fallback is not None:
            self.fallback.reset()
        else:
            self.primary.reset()

    @property
    def size(self) -> int:
        """Return the active accumulator sample count."""
        return self.primary.size
