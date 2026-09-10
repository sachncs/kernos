"""Training loop orchestrating continuous updates and refreshes."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from kernos.basis.nystrom import Nystrom
from kernos.basis.whitening import Whitening
from kernos.core.plan import Plan
from kernos.core.rng import Rng
from kernos.core.state import Bundle, Continuous
from kernos.correct.orth import Ridge
from kernos.correct.rbf import Rbf
from kernos.correct.sampler import Sampler
from kernos.embed.linear import Linear
from kernos.embed.projector import Projector
from kernos.fuse.fuse import Fuse
from kernos.fuse.scaler import Scaler
from kernos.loop.callback import Callback
from kernos.loop.outerstep import Outerstep
from kernos.policy.budget import Budget
from kernos.policy.drift import Frobenius
from kernos.policy.policy import Policy
from kernos.policy.refresh import Refresh
from kernos.solver.direct import Direct


class Loop:
    """Main training loop."""

    def __init__(self, plan: Plan, callbacks: list[Callback] | None = None) -> None:
        self.plan = plan
        self.callbacks = callbacks or []
        self.rng = Rng.fix(plan.seed)
        self.whitening = Whitening(plan.stab_tau, plan.stab_alpha, plan.stab_eps)
        self.scaler = Scaler(plan.stab_eps)
        self.sampler = Sampler(plan.amix)
        self.rbf = Rbf(plan.ltau, plan.lk)
        self.solver = Direct(plan.ridge, plan.stab_jitter, plan.stab_jitter_retry, 10.0, plan.stab_jitter_max, plan.stab_kappa)
        self.fuse = Fuse()
        self.refresh_pipe = Refresh(plan, self.whitening, self.scaler, self.sampler, self.rbf, self.fuse, self.solver)
        self.outerstep = Outerstep(plan)
        self.orth = Ridge(plan.stab_eta)
        self.drift_metric = Frobenius()
        self.budget = Budget(plan.budget)
        self.Rref: np.ndarray | None = None

    def initialize(self, X: np.ndarray, y: np.ndarray) -> Bundle:
        """Initialize state from data and solve for the first ridge coefficients."""
        _, input_dim = X.shape
        embed = Linear(input_dim, self.plan.dim, self.rng)
        R = np.eye(self.plan.dim)
        continuous = Continuous(theta=embed, R=R)
        U = Projector(R).forward(embed.forward(X))
        discrete = self.refresh_pipe.run(Bundle(continuous=continuous, step=0), U, y, self.rng)
        bundle = Bundle(continuous=continuous, discrete=discrete, step=0)
        _, phi, _, _ = self.features(bundle, X)
        weights = self.solver.solve(phi, y)
        return bundle.replace(weights=weights)

    def features(self, bundle: Bundle, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build features and return ``(U, phi, phig, phil)``.

        Public so ``Outerstep.step`` doesn't reach into private API.
        """
        embed = bundle.continuous.theta
        if embed is None:
            raise RuntimeError("Embed not found")
        R = bundle.continuous.R
        if R is None:
            raise RuntimeError("R not initialized")
        proj = Projector(R)
        U = proj.forward(embed.forward(X))
        basis = Nystrom(bundle.discrete.landmarks, bundle.discrete.whitening)
        phig = basis.forward(U)
        phil = self.rbf.forward(U, bundle.discrete.anchors) if bundle.discrete.anchors is not None else np.zeros((X.shape[0], 0))
        if not self.plan.noorth and phil.size > 0:
            phil = self.orth.forward(phig, phil)
        phi = self.fuse.forward(phig, phil, cglobal=bundle.discrete.cglobal, clocal=bundle.discrete.clocal, gateval=bundle.discrete.gateval)
        return U, phi, phig, phil

    def step(self, bundle: Bundle, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> Bundle:
        """Single training step: bump step, continuous update, possibly refresh."""
        new_step = bundle.step + 1
        bundle = bundle.replace(step=new_step)
        indices = self.rng.integers(0, X.shape[0], size=min(self.plan.batch, X.shape[0]))
        X_batch = X[indices]
        y_batch = y[indices]

        def feature_fn(R: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            tmp = bundle.replace(continuous=replace(bundle.continuous, R=R))
            return self.features(tmp, X_batch)[1:]

        bundle = self.outerstep.step(bundle, X_batch, y_batch, feature_fn, self.rng)
        for cb in self.callbacks:
            cb.onstep(new_step, bundle)

        if X_val is not None and y_val is not None:
            bundle = self.maybe_refresh(bundle, X_val, y_val)
        return bundle

    def maybe_refresh(self, bundle: Bundle, X_val: np.ndarray, y_val: np.ndarray) -> Bundle:
        """Evaluate the refresh policy and run ``Refresh`` if it triggers.

        The validation gain (drop in RMSE on ``X_val`` after the
        refresh) is computed and threaded into ``Policy.decide`` so the
        ``gain`` threshold configured on ``Plan`` is actually enforced.
        Refreshes that fail to beat the threshold are discarded.
        """
        if X_val.shape[0] < self.plan.mbasis:
            return bundle
        if self.Rref is None:
            drift_value = 0.001 * bundle.step
        else:
            drift_value = self.drift_metric.measure(bundle.continuous.R, self.Rref)
        baseline_rmse = self._val_rmse(bundle, X_val, y_val)
        candidate = self._apply_refresh(bundle, X_val, y_val)
        new_rmse = self._val_rmse(candidate, X_val, y_val)
        gain = baseline_rmse - new_rmse
        policy = Policy(self.plan, drift_value)
        if policy.decide(candidate, gain=gain) and self.budget.can(self.plan.rcost):
            bundle = candidate
            self.budget.spend(self.plan.rcost)
            self.Rref = bundle.continuous.R.copy()
            for cb in self.callbacks:
                cb.onrefresh(bundle.step, bundle)
        return bundle

    def _apply_refresh(self, bundle: Bundle, X_val: np.ndarray, y_val: np.ndarray) -> Bundle:
        """Run the refresh pipeline on ``(X_val, y_val)`` and solve for weights."""
        U_val = Projector(bundle.continuous.R).forward(bundle.continuous.theta.forward(X_val))
        new_disc = self.refresh_pipe.run(bundle, U_val, y_val, self.rng)
        new_disc = replace(new_disc, tlast=bundle.step)
        candidate = bundle.replace(discrete=new_disc)
        _, phi, _, _ = self.features(candidate, X_val)
        weights = self.solver.solve(phi, y_val)
        return candidate.replace(weights=weights)

    def _val_rmse(self, bundle: Bundle, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """RMSE of the current weights on ``(X_val, y_val)``."""
        _, phi, _, _ = self.features(bundle, X_val)
        pred = phi @ bundle.weights if bundle.weights is not None else np.zeros(X_val.shape[0])
        return float(np.sqrt(np.mean((y_val - pred) ** 2)))
