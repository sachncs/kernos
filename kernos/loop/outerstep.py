"""Outer-step: FD gradient descent on the projection matrix ``R``."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from kernos.core.plan import Plan
from kernos.core.state import Bundle
from kernos.loop.loss import outer
from kernos.solver.direct import Direct


class Outerstep:
    """Simultaneous-perturbation FD optimizer for ``R``."""

    def __init__(self, plan: Plan) -> None:
        self.plan = plan

    def evaluate(
        self, R: np.ndarray, bundle: Bundle, X_batch: np.ndarray, y_batch: np.ndarray, features
    ) -> float:
        """Evaluate the outer objective for a candidate ``R``.

        ``features`` is a callable ``features(R) -> (phi, phig, phil)``
        that builds fused, global, and local features for the given
        projection matrix.
        """
        phi, phig, phil = features(R)
        solver = Direct(
            self.plan.ridge,
            self.plan.stab_jitter,
            self.plan.stab_jitter_retry,
            10.0,
            self.plan.stab_jitter_max,
            self.plan.stab_kappa,
        )
        weights = solver.solve(phi, y_batch)
        return outer(
            y_batch, phi, weights, R, phig, phil, wr=self.plan.wr, worth=self.plan.worth, wdiv=self.plan.wdiv
        )

    def step(
        self, bundle: Bundle, X_batch: np.ndarray, y_batch: np.ndarray, features, rng: np.random.Generator
    ) -> Bundle:
        """One gradient-descent step on ``R``."""
        R = bundle.continuous.R
        if R is None:
            raise RuntimeError("R not initialized")
        rows, cols = R.shape
        sweeps = max(1, min(5, rows * cols // 100))
        gradient = np.zeros_like(R)
        for _ in range(sweeps):
            direction = rng.choice([-1.0, 1.0], size=R.shape) / max(np.linalg.norm(R, "fro"), 1.0)
            plus = R + self.plan.fdeps * direction
            minus = R - self.plan.fdeps * direction
            gradient += (
                (
                    self.evaluate(plus, bundle, X_batch, y_batch, features)
                    - self.evaluate(minus, bundle, X_batch, y_batch, features)
                )
                / (2.0 * self.plan.fdeps)
                * direction
            )
        gradient /= sweeps
        new_R = R - self.plan.lr * gradient
        new_cont = replace(bundle.continuous, R=new_R)
        return bundle.replace(continuous=new_cont)
