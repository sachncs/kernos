"""Typed extension contracts for interchangeable Kernos components.

Concrete implementations remain small classes, while these protocols make
the seams explicit for dependency injection, testing and third-party plugins.
Implementations use structural subtyping: inheritance is optional.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

from kernos.core.plan import Plan
from kernos.core.state import Bundle, Discrete
from kernos.core.types import Array


class FeatureMap(Protocol):
    """Map a batch of inputs to explicit features."""

    def forward(self, X: Array) -> Array:
        """Return explicit features for ``X``."""


class WhiteningMap(Protocol):
    """Build the landmark-kernel whitening representation."""

    def kernel(self, Z: Array, gamma: float = 1.0) -> Array:
        """Return the kernel matrix for landmark rows ``Z``."""

    def build(self, W: Array) -> tuple[Array, Array, Array, int]:
        """Return the eigensystem and retained whitening map."""


class Basis(Protocol):
    """Build explicit global features from persisted landmarks."""

    landmarks: Array

    def forward(self, X: Array) -> Array:
        """Return basis features for ``X``."""


class Solver(Protocol):
    """Solve regularized linear systems in feature space."""

    def solve(self, phi: Array, y: Array) -> Array:
        """Return coefficients for the feature matrix ``phi``."""


class ResidualSolver(Solver, Protocol):
    """Solver that can also produce residuals for anchor selection."""

    def residual(self, phi: Array, y: Array) -> Array:
        """Return per-sample residuals."""


class Orthogonalizer(Protocol):
    """Project local features away from the global feature subspace."""

    def forward(self, global_features: Array, local_features: Array) -> Array:
        """Return orthogonalized local features."""


class LocalFeatureMap(Protocol):
    """Map projected samples and anchors to local correction features."""

    def forward(self, U: Array, anchors: Array) -> Array:
        """Return local features for ``U`` and ``anchors``."""


class AnchorSampler(Protocol):
    """Select correction anchors from coverage and residual signals."""

    def pick(
        self,
        U: Array,
        distances: Array,
        residuals: Array,
        abasis: int,
        rng: np.random.Generator,
        noresid: bool = False,
    ) -> Array:
        """Return selected row indices."""


class FeatureFusion(Protocol):
    """Combine global and local feature blocks."""

    def forward(
        self,
        phig: Array,
        phil: Array,
        cglobal: float | None = None,
        clocal: float | None = None,
        gateval: float | None = None,
    ) -> Array:
        """Return the fused feature matrix."""


class RefreshPipeline(Protocol):
    """Rebuild discrete state from the current projected representation."""

    def run(self, bundle: Bundle, U: Array, y: Array, rng: np.random.Generator) -> Discrete:
        """Return a new discrete state snapshot."""


class OuterStep(Protocol):
    """Perform one continuous representation update."""

    def step(
        self,
        bundle: Bundle,
        X_batch: Array,
        y_batch: Array,
        features: Callable[[Array], tuple[Array, Array, Array]],
        rng: np.random.Generator,
        control_check: Callable[[], None] | None = None,
    ) -> Bundle:
        """Return the updated bundle."""


class BudgetAccountant(Protocol):
    """Track refresh expenditure against a configured budget."""

    spent: float

    def can(self, cost: float) -> bool:
        """Return whether ``cost`` can be spent."""

    def spend(self, cost: float) -> None:
        """Record expenditure or raise a budget error."""


class RefreshPolicy(Protocol):
    """Decide whether a candidate refresh should be committed."""

    def decide(self, bundle: Bundle, gain: float) -> bool:
        """Return whether the candidate satisfies the policy."""


RefreshPolicyFactory = Callable[[Plan, float], RefreshPolicy]


class DriftMetric(Protocol):
    """Measure movement between two continuous representations."""

    def measure(self, current: Array, reference: Array) -> float:
        """Return a scalar drift value."""
