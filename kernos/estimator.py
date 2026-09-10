"""Public estimator: ``Kernos``.

A drop-in scikit-learn-compatible estimator with ``fit``, ``predict``,
``score``, and ``partial_fit``.  Built on top of ``Loop`` and
``Refresh``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from kernos.basis.nystrom import Nystrom
from kernos.core.plan import Buffer, Plan
from kernos.core.state import Bundle
from kernos.embed.projector import Projector
from kernos.loop.loop import Loop
from kernos.loop.callback import Callback, Log
from kernos.predict.predict import Predict


class Kernos(BaseEstimator, RegressorMixin):
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
        self.rcost = rcost
        self.val_frac = val_frac
        self.noref = noref
        self.nohyst = nohyst
        self.nocool = nocool
        self.noresid = noresid
        self.noorth = noorth
        self.nodiv = nodiv
        self.nofreeze = nofreeze

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

    def fit(self, X: np.ndarray, y: np.ndarray, callbacks: list[Callback] | None = None) -> "Kernos":
        """Fit the model to ``(X, y)``.

        Splits ``(X, y)`` internally into training and validation
        slices per ``val_frac``.  The validation slice is fed to the
        refresh policy so we never refresh on training data.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (samples, features); got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D (samples,); got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must agree on n_samples; got X.shape[0]={X.shape[0]} vs y.shape[0]={y.shape[0]}")
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

        self.plan_ = self.build_plan()
        cb = list(callbacks) if callbacks else []
        if self.log_every > 0:
            cb.append(Log(log_every=self.log_every))

        self.loop_ = Loop(self.plan_, callbacks=cb)
        bundle = self.loop_.initialize(X_train, y_train)
        for step in range(1, self.plan_.steps + 1):
            bundle = self.loop_.step(bundle, X_train, y_train, X_val=X_val, y_val=y_val)

        self.bundle_ = bundle
        self.n_features_in_ = X.shape[1]
        self.predictor_ = Predict(weights=bundle.weights)
        self.basis_ = Nystrom(bundle.discrete.landmarks, bundle.discrete.whitening)
        self.X_train_ = X_train
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on new data using the fitted basis and ridge weights."""
        check_fitted = getattr(self, "bundle_", None)
        if check_fitted is None:
            raise RuntimeError("Kernos is not fitted yet; call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (samples, features); got shape {X.shape}")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"feature count mismatch: X has {X.shape[1]} feature(s) but Kernos was fitted on {self.n_features_in_}"
            )
        bundle: Bundle = self.bundle_
        loop = self.loop_ if self.loop_ is not None else Loop(self.plan_)
        _, phi, _, _ = loop.features(bundle, X)
        return self.predictor_.forward(phi)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the R² score on ``(X, y)``.

        Returns ``0.0`` when ``y`` is constant: R² is undefined in that
        case and the previous behaviour silently propagated ``nan``,
        which broke downstream comparisons.
        """
        from kernos.bench.metric import r2

        y_arr = np.asarray(y, dtype=np.float64)
        out = r2(y_arr, self.predict(X))
        if not np.isfinite(out):
            return 0.0
        return out

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> "Kernos":
        """Run exactly one continuous update on ``(X, y)``."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (samples, features); got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D (samples,); got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must agree on n_samples; got X.shape[0]={X.shape[0]} vs y.shape[0]={y.shape[0]}")
        if not hasattr(self, "loop_"):
            return self.fit(X, y)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"feature count mismatch: X has {X.shape[1]} feature(s) but Kernos was fitted on {self.n_features_in_}"
            )
        self.bundle_ = self.loop_.step(self.bundle_, X, y)
        return self

    def __getstate__(self) -> dict:
        """Drop non-picklable callback list before pickling."""
        state = self.__dict__.copy()
        if "loop_" in state and state["loop_"] is not None:
            state["loop_"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore from pickle; ``loop_`` will be rebuilt lazily on next ``fit``."""
        self.__dict__.update(state)

    def get_config(self) -> dict:
        """Return a serializable snapshot of all hyperparameters."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_") and k != "loop_" and k != "bundle_" and k != "predictor_" and k != "basis_" and k != "X_train_" and k != "plan_" and k != "n_features_in_"}

    def set_params(self, **params: Any) -> "Kernos":
        """Set hyperparameters and re-run ``Plan`` validation immediately.

        The default ``BaseEstimator.set_params`` only mutates
        attributes, so inconsistent combinations (e.g. ``abasis >=
        mbasis``) would not surface until the next ``fit``.  Build a
        transient ``Plan`` to fail fast on bad config.
        """
        super().set_params(**params)
        self.build_plan()
        return self
