"""Integration tests for Kernos — the sklearn-compatible estimator."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

from kernos import (
    Buffer,
    Kernos,
    Lifecycle,
    MemoryLimitExceeded,
    RecoveryOutcome,
    SolverEvent,
    SolverStage,
    TrainingCancelled,
    TrainingTimeout,
)


class TestKernos:
    def test_public_buffer_export(self) -> None:
        assert Buffer.FULL.value == "full"
        assert SolverEvent.PCG_BREAKDOWN == "pcg_breakdown"

    def test_fit_returns_self(self, synthetic: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = synthetic
        model = Kernos(
            dim=4, mbasis=32, abasis=4, lk=4, steps=3, batch=16, seed=42, drift_hi=10.0, cool=0, warm=0
        )
        result = model.fit(X, y)
        assert result is model
        assert model.diagnostics_.steps == 3
        assert model.diagnostics_.samples_seen == X.shape[0]
        assert model.diagnostics_.invalid_batches == 0
        assert model.diagnostics_.estimated_memory_bytes is not None
        assert model.diagnostics_.memory_mode == "full"
        assert np.isfinite(model.diagnostics_.elapsed_seconds)
        assert model.diagnostics_.last_condition_number is not None
        assert model.diagnostics_.last_solver_info == 0
        assert model.diagnostics_.solver_nonconvergences == 0
        assert model.diagnostics_.solver_fallbacks == 0
        assert model.diagnostics_.last_solver_event is None
        assert model.diagnostics_.last_solver_stage == SolverStage.INITIALIZE
        assert model.diagnostics_.last_solver_recovery is not None
        assert model.diagnostics_.last_solver_recovery.stage == SolverStage.INITIALIZE
        assert model.diagnostics_.last_solver_recovery.outcome == RecoveryOutcome.NOT_NEEDED
        assert model.lifecycle_ is Lifecycle.FITTED

    def test_predict_shape(self, synthetic: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = synthetic
        model = Kernos(
            dim=4, mbasis=32, abasis=4, lk=4, steps=3, batch=16, seed=42, drift_hi=10.0, cool=0, warm=0
        ).fit(X, y)
        out = model.predict(X)
        assert out.shape == (X.shape[0],)

    def test_score_is_r2(self, synthetic: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = synthetic
        model = Kernos(
            dim=4, mbasis=32, abasis=4, lk=4, steps=3, batch=16, seed=42, drift_hi=10.0, cool=0, warm=0
        ).fit(X, y)
        score = model.score(X, y)
        assert np.isfinite(score)

    def test_predict_and_score_reject_non_finite_values(
        self, synthetic: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = synthetic
        model = Kernos(dim=4, mbasis=32, abasis=4, lk=4, steps=1).fit(X, y)
        invalid_X = X[:3].copy()
        invalid_X[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            model.predict(invalid_X)

        invalid_y = y[:3].copy()
        invalid_y[0] = np.inf
        with pytest.raises(ValueError, match="finite"):
            model.score(X[:3], invalid_y)

        with pytest.raises(ValueError, match="1-D"):
            model.score(X[:3], y[:3, None])

    def test_training_work_is_bounded_by_plan(self, synthetic: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = synthetic
        model = Kernos(
            dim=4,
            mbasis=16,
            abasis=4,
            lk=4,
            steps=5,
            eval_every=2,
            val_frac=0.5,
            batch=16,
            seed=42,
            drift_hi=10.0,
            cool=0,
            warm=0,
            budget=1.0,
            rcost=1.0,
        ).fit(X, y)
        assert model.diagnostics_.steps == 5
        assert model.diagnostics_.evaluations == 2
        assert model.diagnostics_.budget_spent <= model.budget

    def test_refresh_solver_stage_is_observable(self, synthetic: tuple[np.ndarray, np.ndarray]) -> None:
        """Record refresh-stage solves when validation evaluation runs."""
        X, y = synthetic
        model = Kernos(
            dim=4,
            mbasis=16,
            abasis=4,
            lk=4,
            steps=1,
            eval_every=1,
            val_frac=0.5,
            batch=16,
            seed=42,
            drift_hi=1e-9,
            cool=0,
            warm=0,
            budget=1.0,
            rcost=1.0,
        ).fit(X, y)
        assert model.diagnostics_.evaluations == 1
        assert model.diagnostics_.last_solver_recovery is not None
        assert model.diagnostics_.last_solver_recovery.stage == SolverStage.REFRESH
        assert model.diagnostics_.last_solver_stage == SolverStage.REFRESH

    def test_full_and_stream_modes_have_prediction_parity(
        self, synthetic: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = synthetic
        kwargs = {
            "dim": 4,
            "mbasis": 16,
            "abasis": 4,
            "lk": 4,
            "steps": 1,
            "batch": 16,
            "seed": 42,
            "drift_hi": 10.0,
            "cool": 0,
            "warm": 0,
        }
        full = Kernos(mode=Buffer.FULL.value, **kwargs).fit(X, y)
        streamed = Kernos(mode=Buffer.STREAM.value, **kwargs).fit(X, y)

        np.testing.assert_allclose(full.predict(X), streamed.predict(X), rtol=1e-10, atol=1e-10)
        assert streamed.diagnostics_.memory_mode == Buffer.STREAM.value
        assert streamed.diagnostics_.estimated_memory_bytes < full.diagnostics_.estimated_memory_bytes

    def test_get_set_params(self) -> None:
        model = Kernos(dim=4)
        params = model.get_params()
        assert params["dim"] == 4
        model.set_params(dim=8)
        assert model.dim == 8

    def test_set_params_validates(self) -> None:
        model = Kernos()
        with pytest.raises(ValueError):
            model.set_params(ridge=1e-12)
        with pytest.raises(ValueError):
            model.set_params(mbasis=4, abasis=8)

    def test_set_params_is_transactional(self) -> None:
        model = Kernos(dim=4, mbasis=16, abasis=4)
        with pytest.raises(ValueError):
            model.set_params(dim=8, mbasis=4, abasis=8)
        assert model.dim == 4
        assert model.mbasis == 16
        assert model.abasis == 4

    def test_set_params_invalidates_fitted_state(self, synthetic: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = synthetic
        model = Kernos(dim=4, mbasis=32, abasis=4, lk=4, steps=1).fit(X, y)
        model.set_params(ridge=2e-3)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(X)

    def test_clone(self) -> None:
        model = Kernos(dim=4, mbasis=16)
        cloned = clone(model)
        assert cloned.dim == 4
        assert cloned is not model

    def test_pickle_roundtrip(self, synthetic: tuple[np.ndarray, np.ndarray]) -> None:
        import pickle

        X, y = synthetic
        model = Kernos(
            dim=4, mbasis=32, abasis=4, lk=4, steps=3, batch=16, seed=42, drift_hi=10.0, cool=0, warm=0
        ).fit(X, y)
        score_before = model.score(X, y)
        blob = pickle.dumps(model)
        restored = pickle.loads(blob)
        # loop_ is dropped by __getstate__; rebuild via fit on same X
        restored.fit(X, y)
        score_after = restored.score(X, y)
        np.testing.assert_allclose(score_before, score_after, atol=1e-6)

    def test_partial_fit(self, synthetic: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = synthetic
        model = Kernos(
            dim=4, mbasis=32, abasis=4, lk=4, steps=1, batch=16, seed=42, drift_hi=10.0, cool=0, warm=0
        ).fit(X, y)
        result = model.partial_fit(X, y)
        assert result is model
        assert model.lifecycle_ is Lifecycle.STREAMING
        np.testing.assert_allclose(model.predictor_.weights, model.bundle_.weights)
        assert model.diagnostics_.samples_seen == 2 * X.shape[0]
        assert model.diagnostics_.invalid_batches == 0

    def test_cancelled_refit_preserves_previous_snapshot(
        self, synthetic: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = synthetic
        model = Kernos(
            dim=4, mbasis=32, abasis=4, lk=4, steps=1, batch=16, seed=42, drift_hi=10.0, cool=0, warm=0
        ).fit(X, y)
        before_bundle = model.bundle_
        before_prediction = model.predict(X[:5]).copy()
        with pytest.raises(TrainingCancelled):
            model.fit(X, y, cancel_check=lambda: True)
        assert model.lifecycle_ is Lifecycle.FITTED
        assert model.bundle_ is before_bundle
        np.testing.assert_allclose(model.predict(X[:5]), before_prediction)

    def test_pickled_model_can_resume_partial_fit(self, synthetic: tuple[np.ndarray, np.ndarray]) -> None:
        import pickle

        X, y = synthetic
        model = Kernos(
            dim=4, mbasis=32, abasis=4, lk=4, steps=1, batch=16, seed=42, drift_hi=10.0, cool=0, warm=0
        ).fit(X, y)
        step_before = model.bundle_.step
        restored = pickle.loads(pickle.dumps(model))
        restored.partial_fit(X, y)
        assert restored.bundle_.step == step_before + 1
        assert restored.diagnostics_.steps == step_before + 1
        assert restored.bundle_.weights is not None

    def test_n_features_in_recorded(self, synthetic: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = synthetic
        model = Kernos(
            dim=4, mbasis=32, abasis=4, lk=4, steps=1, batch=16, seed=42, drift_hi=10.0, cool=0, warm=0
        ).fit(X, y)
        assert model.n_features_in_ == X.shape[1]

    def test_predict_before_fit_raises(self) -> None:
        model = Kernos()
        assert model.lifecycle_ is Lifecycle.UNFITTED
        with pytest.raises(RuntimeError):
            model.predict(np.zeros((5, 4)))

    def test_fit_too_few_samples_raises(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((5, 3))
        y = rng.standard_normal(5)
        model = Kernos(dim=4, mbasis=16, abasis=4, lk=2, steps=1)
        with pytest.raises(ValueError, match="n_samples"):
            model.fit(X, y)

    def test_non_finite_inputs_raise(self) -> None:
        X = np.ones((20, 3))
        y = np.ones(20)
        X[0, 0] = np.nan
        model = Kernos(dim=4, mbasis=8, abasis=2, lk=2, steps=1)
        with pytest.raises(ValueError, match="finite"):
            model.fit(X, y)
        X[0, 0] = 1.0
        model.fit(X, y)
        assert model.diagnostics_.invalid_batches == 1
        assert model.diagnostics_.invalid_samples == 20

    def test_memory_budget_fails_before_fit(self, synthetic: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = synthetic
        model = Kernos(dim=4, mbasis=32, abasis=4, lk=4, steps=1, memory_limit_mb=0.001)
        with pytest.raises(MemoryLimitExceeded, match="memory_limit_mb"):
            model.fit(X, y)

    def test_fit_can_be_cancelled_without_publishing_state(
        self, synthetic: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = synthetic
        checks = 0

        def cancel() -> bool:
            nonlocal checks
            checks += 1
            return checks >= 1

        model = Kernos(dim=4, mbasis=32, abasis=4, lk=4, steps=3)
        with pytest.raises(TrainingCancelled):
            model.fit(X, y, cancel_check=cancel)
        assert not hasattr(model, "bundle_")

    def test_partial_fit_cancellation_preserves_bundle(
        self, synthetic: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = synthetic
        model = Kernos(dim=4, mbasis=32, abasis=4, lk=4, steps=1).fit(X, y)
        before = model.bundle_
        with pytest.raises(TrainingCancelled):
            model.partial_fit(X, y, cancel_check=lambda: True)
        assert model.bundle_ is before

    def test_rejected_streaming_batch_is_observable_and_transactional(
        self, synthetic: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = synthetic
        model = Kernos(dim=4, mbasis=32, abasis=4, lk=4, steps=1).fit(X, y)
        before = model.bundle_
        invalid_X = X[:3].copy()
        invalid_X[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            model.partial_fit(invalid_X, y[:3])
        assert model.bundle_ is before
        assert model.diagnostics_.invalid_batches == 1
        assert model.diagnostics_.invalid_samples == 3
        assert model.diagnostics_.samples_seen == X.shape[0]

    def test_timeout_is_validated_and_preserves_state(self, synthetic: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = synthetic
        model = Kernos(dim=4, mbasis=32, abasis=4, lk=4, steps=1).fit(X, y)
        before = model.bundle_
        with pytest.raises(ValueError, match="timeout_seconds"):
            model.partial_fit(X, y, timeout_seconds=0.0)
        with pytest.raises(TrainingTimeout):
            model.partial_fit(X, y, timeout_seconds=1e-12)
        assert model.bundle_ is before

    def test_default_hyperparameters(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((1200, 3))
        y = X[:, 0] + 0.1 * rng.standard_normal(1200)
        model = Kernos()
        model.fit(X, y)
        params = model.get_params()
        assert params["dim"] == 64
        assert params["mbasis"] == 512
        assert params["ridge"] == pytest.approx(1e-3)
        pred = model.predict(X[:5])
        assert pred.shape == (5,)
        assert np.isfinite(model.score(X[:200], y[:200]))

    def test_default_modes(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((1200, 3))
        y = X[:, 0] + 0.1 * rng.standard_normal(1200)
        for mode in ("full", "stream", "adaptive"):
            model = Kernos(dim=8, mbasis=16, abasis=2, lk=2, steps=1, mode=mode)
            model.fit(X, y)
            assert np.isfinite(model.score(X[:200], y[:200]))

    def test_alternative_components(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((1200, 3))
        y = X[:, 0] + 0.1 * rng.standard_normal(1200)
        model = Kernos(
            dim=8,
            mbasis=16,
            abasis=2,
            lk=2,
            steps=1,
            embedder="identity",
            basis="greedy",
            solver="iterative",
            drift="spectral",
        )
        model.fit(X, y)
        assert np.isfinite(model.score(X[:200], y[:200]))
