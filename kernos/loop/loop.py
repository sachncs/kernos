"""Training loop orchestrating continuous updates and refreshes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from time import monotonic

import numpy as np

from kernos.core.composition import LoopComponents, build_loop_components
from kernos.core.diagnostics import RecoveryOutcome, SolverRecovery, SolverStage, TrainingDiagnostics
from kernos.core.factory import build_embedder, restore_basis
from kernos.core.interfaces import Basis, FeatureMap
from kernos.core.plan import Plan
from kernos.core.rng import Rng
from kernos.core.state import Bundle, Continuous, Discrete
from kernos.embed.projector import Projector
from kernos.loop.callback import Callback


class Loop:
    """Main training loop."""

    def __init__(
        self,
        plan: Plan,
        callbacks: list[Callback] | None = None,
        components: LoopComponents | None = None,
    ) -> None:
        """Initialize the training pipeline from an immutable plan.

        Args:
            plan: Validated immutable training configuration.
            callbacks: Optional lifecycle callbacks.
            components: Optional injected runtime graph. When omitted, the
                registered default factories build one from ``plan``.
        """
        self.plan = plan
        self.callbacks = callbacks or []
        self.rng = Rng.fix(plan.seed)
        graph = components or build_loop_components(plan)
        self._whitening = graph.whitening
        self._scaler = graph.scaler
        self._sampler = graph.sampler
        self._rbf = graph.rbf
        self._solver = graph.solver
        self._fuse = graph.fuse
        self._refresh_pipe = graph.refresh_pipe
        self._outerstep = graph.outerstep
        self._orth = graph.orth
        self._drift_metric = graph.drift_metric
        self._budget = graph.budget
        self._policy_factory = graph.policy_factory
        self._r_ref: np.ndarray | None = None
        self._started_at = monotonic()
        self.diagnostics = TrainingDiagnostics(memory_mode=plan.mode.value)

    def _update_elapsed(self) -> None:
        """Publish elapsed wall-clock time without mutating a snapshot."""
        self.diagnostics = self.diagnostics.replace(elapsed_seconds=max(0.0, monotonic() - self._started_at))

    def initialize(
        self, X: np.ndarray, y: np.ndarray, control_check: Callable[[], None] | None = None
    ) -> Bundle:
        """Initialize state from data and solve for the first ridge coefficients."""
        if control_check is not None:
            control_check()
        _, input_dim = X.shape
        embed = self._build_embed(input_dim)
        R = self._initial_R(input_dim, X)
        continuous = Continuous(theta=embed, R=R)
        U = Projector(R).forward(embed.forward(X))
        discrete = self._refresh_pipe.run(Bundle(continuous=continuous, step=0), U, y, self.rng)
        if control_check is not None:
            control_check()
        bundle = Bundle(continuous=continuous, discrete=discrete, step=0)
        _, phi, _, _ = self.features(bundle, X)
        try:
            weights = self._solver.solve(phi, y)
        except BaseException:
            self._record_solver_status(SolverStage.INITIALIZE, failed=True)
            raise
        if control_check is not None:
            control_check()
        self._record_solver_status(SolverStage.INITIALIZE)
        self.diagnostics = self.diagnostics.replace(
            estimated_memory_bytes=self.plan.estimate_memory_bytes(X.shape[0], X.shape[1])
        )
        self._update_elapsed()
        return bundle.replace(weights=weights)

    def _build_embed(self, input_dim: int) -> FeatureMap:
        """Build the configured embedder through the component registry."""
        return build_embedder(self.plan.embedder, input_dim, self.plan.dim, self.rng)

    def _record_solver_status(self, stage: SolverStage, failed: bool = False) -> None:
        """Copy optional built-in solver status into immutable diagnostics."""
        info = int(getattr(self._solver, "last_info", 0))
        iterations = int(getattr(self._solver, "last_iterations", 0))
        event = getattr(self._solver, "last_solver_event", None)
        jitter = getattr(self._solver, "last_jitter", None)
        condition_number = getattr(self._solver, "last_condition", None)
        nonconvergences = self.diagnostics.solver_nonconvergences + int(info != 0)
        fallbacks = self.diagnostics.solver_fallbacks + int(event is not None)
        if failed:
            outcome = RecoveryOutcome.FAILED
        elif info != 0:
            outcome = RecoveryOutcome.NONCONVERGED
        elif event is not None:
            outcome = RecoveryOutcome.RECOVERED
        else:
            outcome = RecoveryOutcome.NOT_NEEDED
        recovery = SolverRecovery(
            stage=stage,
            outcome=outcome,
            event=event,
            attempts=int(getattr(self._solver, "last_attempts", int(bool(event and jitter is not None)))),
            iterations=iterations,
            info=info,
            jitter=jitter,
            condition_number=condition_number,
        )
        self.diagnostics = self.diagnostics.replace(
            solver_iterations=iterations,
            solver_nonconvergences=nonconvergences,
            last_solver_info=info,
            solver_fallbacks=fallbacks,
            last_solver_event=event,
            last_solver_jitter=jitter,
            last_condition_number=condition_number,
            last_solver_stage=stage,
            last_solver_recovery=recovery,
        )

    def _initial_R(self, input_dim: int, X: np.ndarray) -> np.ndarray:
        """Build the initial projection matrix ``R``.

        For ``Linear`` we use ``plan.dim`` so the projected dimension
        matches the embedder.  For ``Identity`` the projector is the
        identity on ``input_dim``.
        """
        if self.plan.embedder == "identity":
            return np.eye(input_dim)
        return np.eye(self.plan.dim)

    def features(
        self, bundle: Bundle, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        basis = self._build_basis_from(bundle.discrete)
        phig = basis.forward(U)
        phil = (
            self._rbf.forward(U, bundle.discrete.anchors)
            if bundle.discrete.anchors is not None
            else np.zeros((X.shape[0], 0))
        )
        if not self.plan.noorth and phil.size > 0:
            phil = self._orth.forward(phig, phil)
        phi = self._fuse.forward(
            phig,
            phil,
            cglobal=bundle.discrete.cglobal,
            clocal=bundle.discrete.clocal,
            gateval=bundle.discrete.gateval,
        )
        return U, phi, phig, phil

    def _build_basis_from(self, discrete: Discrete) -> Basis:
        """Reconstruct the basis object from a stored ``Discrete`` snapshot."""
        if discrete.landmarks is None:
            raise RuntimeError("Persisted basis requires landmarks in Discrete")
        try:
            return restore_basis(self.plan.basis, discrete.landmarks, discrete.whitening, discrete.basis_wz)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

    def step(
        self,
        bundle: Bundle,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        control_check: Callable[[], None] | None = None,
    ) -> Bundle:
        """Single training step: bump step, continuous update, possibly refresh."""
        new_step = bundle.step + 1
        bundle = bundle.replace(step=new_step)
        indices = self.rng.integers(0, X.shape[0], size=min(self.plan.batch, X.shape[0]))
        X_batch = X[indices]
        y_batch = y[indices]

        def feature_fn(R: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            tmp = bundle.replace(continuous=replace(bundle.continuous, R=R))
            return self.features(tmp, X_batch)[1:]

        bundle = self._outerstep.step(bundle, X_batch, y_batch, feature_fn, self.rng, control_check)
        for cb in self.callbacks:
            cb.onstep(new_step, bundle)
        self.diagnostics = self.diagnostics.replace(steps=new_step)
        self._update_elapsed()

        if X_val is not None and y_val is not None and new_step % self.plan.eval_every == 0:
            bundle = self.maybe_refresh(bundle, X_val, y_val, control_check)
        return bundle

    def maybe_refresh(
        self,
        bundle: Bundle,
        X_val: np.ndarray,
        y_val: np.ndarray,
        control_check: Callable[[], None] | None = None,
    ) -> Bundle:
        """Evaluate the refresh policy and run ``Refresh`` if it triggers.

        The validation gain (drop in RMSE on ``X_val`` after the
        refresh) is computed and threaded into ``Policy.decide`` so the
        ``gain`` threshold configured on ``Plan`` is actually enforced.
        Refreshes that fail to beat the threshold are discarded.
        """
        if X_val.shape[0] < self.plan.mbasis:
            return bundle
        if self._r_ref is None:
            drift_value = 0.001 * bundle.step
        else:
            if bundle.continuous.R is None:
                raise RuntimeError("R not initialized")
            drift_value = self._drift_metric.measure(bundle.continuous.R, self._r_ref)
        baseline_rmse = self._val_rmse(bundle, X_val, y_val, control_check)
        candidate = self._apply_refresh(bundle, X_val, y_val, control_check)
        new_rmse = self._val_rmse(candidate, X_val, y_val, control_check)
        gain = baseline_rmse - new_rmse
        self.diagnostics = self.diagnostics.replace(
            evaluations=self.diagnostics.evaluations + 1,
            last_drift=float(drift_value),
            last_gain=float(gain),
        )
        policy = self._policy_factory(self.plan, drift_value)
        if policy.decide(candidate, gain=gain) and self._budget.can(self.plan.refresh_cost):
            bundle = candidate
            self._budget.spend(self.plan.refresh_cost)
            self.diagnostics = self.diagnostics.replace(
                refreshes=self.diagnostics.refreshes + 1,
                budget_spent=self._budget.spent,
            )
            if bundle.continuous.R is None:
                raise RuntimeError("R not initialized")
            self._r_ref = bundle.continuous.R.copy()
            for cb in self.callbacks:
                cb.onrefresh(bundle.step, bundle)
        return bundle

    def _apply_refresh(
        self,
        bundle: Bundle,
        X_val: np.ndarray,
        y_val: np.ndarray,
        control_check: Callable[[], None] | None = None,
    ) -> Bundle:
        """Run the refresh pipeline on ``(X_val, y_val)`` and solve for weights."""
        if bundle.continuous.R is None or bundle.continuous.theta is None:
            raise RuntimeError("continuous state is not initialized")
        U_val = Projector(bundle.continuous.R).forward(bundle.continuous.theta.forward(X_val))
        if control_check is not None:
            control_check()
        new_disc = self._refresh_pipe.run(bundle, U_val, y_val, self.rng)
        new_disc = replace(new_disc, tlast=bundle.step)
        candidate = bundle.replace(discrete=new_disc)
        _, phi, _, _ = self.features(candidate, X_val)
        try:
            weights = self._solver.solve(phi, y_val)
        except BaseException:
            self._record_solver_status(SolverStage.REFRESH, failed=True)
            raise
        if control_check is not None:
            control_check()
        self._record_solver_status(SolverStage.REFRESH)
        return candidate.replace(weights=weights)

    def _val_rmse(
        self,
        bundle: Bundle,
        X_val: np.ndarray,
        y_val: np.ndarray,
        control_check: Callable[[], None] | None = None,
    ) -> float:
        """RMSE of the current weights on ``(X_val, y_val)``."""
        if control_check is not None:
            control_check()
        _, phi, _, _ = self.features(bundle, X_val)
        pred = phi @ bundle.weights if bundle.weights is not None else np.zeros(X_val.shape[0])
        return float(np.sqrt(np.mean((y_val - pred) ** 2)))
