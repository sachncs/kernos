"""Public estimator: ``Kernos``.

A scikit-learn-style estimator with ``fit``, ``predict``,
``score``, and ``partial_fit``.  Built on top of ``Loop`` and
``Refresh``.
"""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from numbers import Real
from time import monotonic
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from kernos.basis.greedy import Greedy
from kernos.basis.nystrom import Nystrom
from kernos.core.error import MemoryLimitExceeded, TrainingCancelled, TrainingTimeout
from kernos.core.lifecycle import Lifecycle
from kernos.core.plan import Buffer, Plan
from kernos.core.state import Bundle
from kernos.loop.callback import Callback, Log
from kernos.loop.loop import Loop
from kernos.predict.predict import Predict


class Kernos(BaseEstimator, RegressorMixin):  # type: ignore[misc]
    """Refresh-aware hybrid kernel regressor."""

    def __init__(
        self,
        dim: int = 64,
        mbasis: int = 512,
        abasis: int = 128,
        ridge: float = 1e-3,
        mode: str = "full",
        embedder: str = "linear",
        basis: str = "nystrom",
        solver: str = "direct",
        drift: str = "frobenius",
        steps: int = 1000,
        eval_every: int = 10,
        batch: int = 32,
        seed: int | None = None,
        drift_hi: float = 0.1,
        cool: int = 50,
        warm: int = 10,
        gain: float = 0.01,
        amix: float = 0.5,
        ltau: float = 0.1,
        lk: int = 5,
        log_every: int = 0,
        lr: float = 1e-4,
        wr: float = 0.0,
        worth: float = 0.0,
        wdiv: float = 0.0,
        fdeps: float = 1e-5,
        budget: float = float("inf"),
        memory_limit_mb: float | None = None,
        rcost: float = 1.0,
        val_frac: float = 0.15,
        noref: bool = False,
        nohyst: bool = False,
        nocool: bool = False,
        noresid: bool = False,
        noorth: bool = False,
        nodiv: bool = False,
        nofreeze: bool = False,
    ) -> None:
        """Initialize the scikit-learn-style estimator configuration."""
        self.dim = dim
        self.mbasis = mbasis
        self.abasis = abasis
        self.ridge = ridge
        self.mode = mode
        self.embedder = embedder
        self.basis = basis
        self.solver = solver
        self.drift = drift
        self.steps = steps
        self.eval_every = eval_every
        self.batch = batch
        self.seed = seed
        self.drift_hi = drift_hi
        self.cool = cool
        self.warm = warm
        self.gain = gain
        self.amix = amix
        self.ltau = ltau
        self.lk = lk
        self.log_every = log_every
        self.lr = lr
        self.wr = wr
        self.worth = worth
        self.wdiv = wdiv
        self.fdeps = fdeps
        self.budget = budget
        self.memory_limit_mb = memory_limit_mb
        self.rcost = rcost
        self.val_frac = val_frac
        self.noref = noref
        self.nohyst = nohyst
        self.nocool = nocool
        self.noresid = noresid
        self.noorth = noorth
        self.nodiv = nodiv
        self.nofreeze = nofreeze
        self._lifecycle = Lifecycle.UNFITTED
        self._invalid_batches = 0
        self._invalid_samples = 0

    @property
    def lifecycle_(self) -> Lifecycle:
        """Return the explicit estimator lifecycle state."""
        return self._lifecycle

    def build_plan(self) -> Plan:
        """Translate sklearn-style kwargs to a frozen :class:`Plan`."""
        buffer = Buffer(self.mode) if isinstance(self.mode, str) else self.mode
        return Plan(
            dim=self.dim,
            mbasis=self.mbasis,
            abasis=self.abasis,
            ridge=self.ridge,
            mode=buffer,
            embedder=self.embedder,
            basis=self.basis,
            solver=self.solver,
            drift=self.drift,
            steps=self.steps,
            eval_every=self.eval_every,
            batch=self.batch,
            seed=self.seed,
            drift_hi=self.drift_hi,
            cool=self.cool,
            warm=self.warm,
            gain=self.gain,
            amix=self.amix,
            ltau=self.ltau,
            lk=self.lk,
            log_every=self.log_every,
            lr=self.lr,
            wr=self.wr,
            worth=self.worth,
            wdiv=self.wdiv,
            fdeps=self.fdeps,
            budget=self.budget,
            memory_limit_mb=self.memory_limit_mb,
            rcost=self.rcost,
            val_frac=self.val_frac,
            noref=self.noref,
            nohyst=self.nohyst,
            nocool=self.nocool,
            noresid=self.noresid,
            noorth=self.noorth,
            nodiv=self.nodiv,
            nofreeze=self.nofreeze,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        callbacks: list[Callback] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        timeout_seconds: float | None = None,
    ) -> Kernos:
        """Fit the model to ``(X, y)``.

        Splits ``(X, y)`` internally into training and validation
        slices per ``val_frac``.  The validation slice is fed to the
        refresh policy so we never refresh on training data.
        """
        previous_lifecycle = self._lifecycle
        self._lifecycle = Lifecycle.FITTING
        try:
            try:
                X, y = self._validate_inputs(X, y)
            except (TypeError, ValueError):
                self._record_invalid_batch(X)
                raise
            deadline = self._training_deadline(timeout_seconds)
            self._check_memory_budget(X.shape[0], X.shape[1])
            min_required = int(np.ceil(self.mbasis / max(1.0 - self.val_frac, 1e-9))) + 2
            if X.shape[0] < min_required:
                raise ValueError(
                    f"n_samples ({X.shape[0]}) must be >= {min_required} to support "
                    f"mbasis={self.mbasis} with val_frac={self.val_frac}; lower mbasis or supply more data"
                )
            n = X.shape[0]
            n_val = max(1, int(self.val_frac * n))
            n_train = n - n_val
            rng = np.random.default_rng(self.seed)
            perm = rng.permutation(n)
            train_idx = perm[:n_train]
            val_idx = perm[n_train:]
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            plan = self.build_plan()
            cb = list(callbacks) if callbacks else []
            if self.log_every > 0:
                cb.append(Log(log_every=self.log_every))

            loop = Loop(plan, callbacks=cb)

            def control_check() -> None:
                self._check_training_control(cancel_check, deadline)

            bundle = loop.initialize(X_train, y_train, control_check=control_check)
            self._check_training_control(cancel_check, deadline)
            for _step in range(1, plan.steps + 1):
                self._check_training_control(cancel_check, deadline)
                bundle = loop.step(
                    bundle,
                    X_train,
                    y_train,
                    X_val=X_val,
                    y_val=y_val,
                    control_check=control_check,
                )

            # Commit all fitted state only after the complete operation succeeds.
            self.plan_ = plan
            self.loop_ = loop
            self.bundle_ = bundle
            self.n_features_in_ = X.shape[1]
            self._sync_runtime_state()
            self.diagnostics_ = loop.diagnostics.replace(
                samples_seen=int(X.shape[0]),
                invalid_batches=self._invalid_batches,
                invalid_samples=self._invalid_samples,
            )
            loop.diagnostics = self.diagnostics_
            self.X_train_ = X_train
            self._lifecycle = Lifecycle.FITTED
            return self
        except BaseException:
            self._lifecycle = previous_lifecycle
            raise

    @staticmethod
    def _validate_inputs(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Normalize and validate a supervised regression batch."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2-D (samples, features); got shape {X_arr.shape}")
        if y_arr.ndim != 1:
            raise ValueError(f"y must be 1-D (samples,); got shape {y_arr.shape}")
        if X_arr.shape[0] == 0:
            raise ValueError("X and y must contain at least one sample")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X and y must agree on n_samples; got X.shape[0]={X_arr.shape[0]} vs y.shape[0]={y_arr.shape[0]}"
            )
        if not np.isfinite(X_arr).all() or not np.isfinite(y_arr).all():
            raise ValueError("X and y must contain only finite values")
        return X_arr, y_arr

    def _sync_runtime_state(self) -> None:
        """Refresh derived runtime objects from the current immutable bundle."""
        bundle: Bundle = self.bundle_
        if bundle.weights is None:
            raise RuntimeError("Kernos state has no solved weights")
        self.predictor_ = Predict(weights=bundle.weights)
        self.basis_: Nystrom | Greedy
        if self.plan_.basis == "greedy":
            if bundle.discrete.landmarks is None or bundle.discrete.basis_wz is None:
                raise RuntimeError("Greedy basis state is incomplete")
            self.basis_ = Greedy(bundle.discrete.landmarks, bundle.discrete.basis_wz)
        else:
            if bundle.discrete.landmarks is None or bundle.discrete.whitening is None:
                raise RuntimeError("Nystrom basis state is incomplete")
            self.basis_ = Nystrom(bundle.discrete.landmarks, bundle.discrete.whitening)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on new data using the fitted basis and ridge weights."""
        if self._lifecycle not in {Lifecycle.FITTED, Lifecycle.STREAMING}:
            raise RuntimeError("Kernos is not fitted yet; call fit() first.")
        X = self._validate_predict_features(X)
        bundle: Bundle = self.bundle_
        loop = self.loop_ if self.loop_ is not None else Loop(self.plan_)
        _, phi, _, _ = loop.features(bundle, X)
        if bundle.weights is None:
            raise RuntimeError("Kernos is fitted but has no solved weights")
        return Predict(weights=bundle.weights).forward(phi)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the R² score on ``(X, y)``.

        Returns ``0.0`` when ``y`` is constant: R² is undefined in that
        case and the previous behaviour silently propagated ``nan``,
        which broke downstream comparisons.
        """
        from kernos.bench.metric import r2

        X_arr, y_arr = self._validate_inputs(X, y)
        out = r2(y_arr, self.predict(X_arr))
        if not np.isfinite(out):
            return 0.0
        return out

    def _validate_predict_features(self, X: np.ndarray) -> np.ndarray:
        """Normalize prediction features and reject invalid values."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2-D (samples, features); got shape {X_arr.shape}")
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"feature count mismatch: X has {X_arr.shape[1]} feature(s) but Kernos was fitted on {self.n_features_in_}"
            )
        if not np.isfinite(X_arr).all():
            raise ValueError("X must contain only finite values")
        return X_arr

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cancel_check: Callable[[], bool] | None = None,
        timeout_seconds: float | None = None,
    ) -> Kernos:
        """Run exactly one continuous update on ``(X, y)``."""
        try:
            X, y = self._validate_inputs(X, y)
        except (TypeError, ValueError):
            self._record_invalid_batch(X)
            raise
        deadline = self._training_deadline(timeout_seconds)
        self._check_memory_budget(X.shape[0], X.shape[1])
        if not hasattr(self, "bundle_") or not hasattr(self, "plan_"):
            self._check_training_control(cancel_check, deadline)
            result = self.fit(X, y, cancel_check=cancel_check, timeout_seconds=timeout_seconds)
            self._lifecycle = Lifecycle.STREAMING
            return result
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"feature count mismatch: X has {X.shape[1]} feature(s) but Kernos was fitted on {self.n_features_in_}"
            )
        # ``__getstate__`` intentionally removes callbacks and the live loop.
        # Recreate the loop around the persisted plan/state so a pickled model
        # continues streaming instead of silently starting a new fit.
        if self.loop_ is None:
            self.loop_ = Loop(self.plan_)
            if hasattr(self, "diagnostics_"):
                self.loop_.diagnostics = self.diagnostics_
        self._check_training_control(cancel_check, deadline)

        def control_check() -> None:
            self._check_training_control(cancel_check, deadline)

        self.bundle_ = self.loop_.step(self.bundle_, X, y, control_check=control_check)
        self._sync_runtime_state()
        self.diagnostics_ = self.loop_.diagnostics.replace(
            samples_seen=self.loop_.diagnostics.samples_seen + int(X.shape[0])
        )
        self.loop_.diagnostics = self.diagnostics_
        self._lifecycle = Lifecycle.STREAMING
        return self

    @staticmethod
    def _training_deadline(timeout_seconds: float | None) -> float | None:
        """Return a monotonic deadline after validating an optional timeout."""
        if timeout_seconds is None:
            return None
        if (
            not isinstance(timeout_seconds, Real)
            or isinstance(timeout_seconds, bool)
            or not isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a finite positive number or None")
        return monotonic() + float(timeout_seconds)

    @staticmethod
    def _check_training_control(cancel_check: Callable[[], bool] | None, deadline: float | None) -> None:
        """Stop training cooperatively when cancellation or timeout is requested."""
        if cancel_check is not None and cancel_check():
            raise TrainingCancelled("training cancelled at a step boundary")
        if deadline is not None and monotonic() >= deadline:
            raise TrainingTimeout("training timeout expired at a step boundary")

    def _check_memory_budget(self, n_samples: int, n_features: int) -> None:
        """Raise before numerical work when the configured memory budget is exceeded."""
        if self.memory_limit_mb is None:
            return
        estimated = self.build_plan().estimate_memory_bytes(n_samples, n_features)
        limit = self.memory_limit_mb * 1024 * 1024
        if estimated > limit:
            raise MemoryLimitExceeded(
                f"estimated working set ({estimated / 1024**2:.1f} MiB) exceeds "
                f"memory_limit_mb={self.memory_limit_mb:.1f}"
            )

    def _record_invalid_batch(self, X: np.ndarray) -> None:
        """Record a rejected streaming batch without mutating model state."""
        try:
            candidate = np.asarray(X)
            sample_count = int(candidate.shape[0]) if candidate.ndim >= 1 else 0
        except (TypeError, ValueError):
            sample_count = 0
        self._invalid_batches += 1
        self._invalid_samples += sample_count
        if not hasattr(self, "diagnostics_"):
            return
        self.diagnostics_ = self.diagnostics_.replace(
            invalid_batches=self._invalid_batches,
            invalid_samples=self._invalid_samples,
        )
        if getattr(self, "loop_", None) is not None:
            self.loop_.diagnostics = self.diagnostics_

    def __getstate__(self) -> dict[str, Any]:
        """Drop non-picklable callback list before pickling."""
        state = self.__dict__.copy()
        if "loop_" in state and state["loop_"] is not None:
            state["loop_"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore from pickle; ``loop_`` will be rebuilt lazily on next ``fit``."""
        self.__dict__.update(state)
        if "_lifecycle" not in self.__dict__:
            self._lifecycle = Lifecycle.FITTED if "bundle_" in state else Lifecycle.UNFITTED

    def get_config(self) -> dict[str, Any]:
        """Return a serializable snapshot of all hyperparameters."""
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_")
            and k != "loop_"
            and k != "bundle_"
            and k != "predictor_"
            and k != "basis_"
            and k != "X_train_"
            and k != "plan_"
            and k != "n_features_in_"
        }

    def set_params(self, **params: Any) -> Kernos:
        """Set hyperparameters and re-run ``Plan`` validation immediately.

        The default ``BaseEstimator.set_params`` only mutates
        attributes, so inconsistent combinations (e.g. ``abasis >=
        mbasis``) would not surface until the next ``fit``.  Build a
        transient ``Plan`` to fail fast on bad config.
        """
        candidate = self.get_params(deep=False)
        unknown = sorted(set(params) - set(candidate))
        if unknown:
            raise ValueError(f"Invalid parameter(s) for Kernos: {', '.join(unknown)}")
        candidate.update(params)
        # Validate a complete candidate before mutating this estimator.  This
        # makes failed reconfiguration transactional.
        candidate_model = type(self)(**candidate)
        candidate_model.build_plan()
        super().set_params(**params)
        self._invalidate_fitted_state()
        return self

    def _invalidate_fitted_state(self) -> None:
        """Remove derived fit state after a validated hyperparameter change."""
        for name in (
            "bundle_",
            "basis_",
            "diagnostics_",
            "loop_",
            "n_features_in_",
            "plan_",
            "predictor_",
            "X_train_",
        ):
            self.__dict__.pop(name, None)
        self._lifecycle = Lifecycle.UNFITTED
