"""Outer-step: FD gradient descent on the projection matrix ``R``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from kernos.core.plan import Plan
from kernos.core.state import Bundle
from kernos.loop.loss import outer
from kernos.solver.direct import Direct


class Outerstep:
    """Simultaneous-perturbation FD optimizer for ``R``."""

    def __init__(self, plan: Plan) -> None:
        """Initialize finite-difference optimization from a training plan."""
        self._plan = plan

    def evaluate(
        self,
        R: np.ndarray,
        bundle: Bundle,
        X_batch: np.ndarray,
        y_batch: np.ndarray,
        features: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
        control_check: Callable[[], None] | None = None,
    ) -> float:
        """Evaluate the outer objective for a candidate ``R``.

        ``features`` is a callable ``features(R) -> (phi, phig, phil)``
        that builds fused, global, and local features for the given
        projection matrix.
        """
        if control_check is not None:
            control_check()
        phi, phig, phil = features(R)
        solver = Direct(
            self._plan.ridge,
            self._plan.stab_jitter,
            self._plan.stab_jitter_retry,
            10.0,
            self._plan.stab_jitter_max,
            self._plan.stab_kappa,
            check_condition=False,
        )
        weights = solver.solve(phi, y_batch)
        if control_check is not None:
            control_check()
        return outer(
            y_batch,
            phi,
            weights,
            R,
            phig,
            phil,
            wr=self._plan.representation_regularization,
            worth=self._plan.orthogonality_regularization,
            wdiv=self._plan.diversity_regularization,
        )

    def step(
        self,
        bundle: Bundle,
        X_batch: np.ndarray,
        y_batch: np.ndarray,
        features: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
        rng: np.random.Generator,
        control_check: Callable[[], None] | None = None,
    ) -> Bundle:
        """One gradient-descent step on ``R``."""
        R = bundle.continuous.R
        if R is None:
            raise RuntimeError("R not initialized")
        rows, cols = R.shape
        sweeps = max(1, min(5, rows * cols // 100))
        gradient = np.zeros_like(R)
        for _ in range(sweeps):
            if control_check is not None:
                control_check()
            direction = rng.choice([-1.0, 1.0], size=R.shape) / max(np.linalg.norm(R, "fro"), 1.0)
            plus = R + self._plan.finite_difference_step * direction
            minus = R - self._plan.finite_difference_step * direction
            gradient += (
                (
                    self.evaluate(plus, bundle, X_batch, y_batch, features, control_check)
                    - self.evaluate(minus, bundle, X_batch, y_batch, features, control_check)
                )
                / (2.0 * self._plan.finite_difference_step)
                * direction
            )
        gradient /= sweeps
        new_R = R - self._plan.lr * gradient
        new_cont = replace(bundle.continuous, R=new_R)
        return bundle.replace(continuous=new_cont)
