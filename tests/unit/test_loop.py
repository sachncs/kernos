"""Unit tests for kernos.loop."""

from __future__ import annotations

import numpy as np
import pytest

from kernos.loop.callback import Log, Profile, Snapshot
from kernos.loop.loop import Loop
from kernos.loop.loss import div, orth, outer, ridge
from kernos.loop.outerstep import Outerstep


class TestLossFunctions:
    def test_ridge(self, rng: np.random.Generator) -> None:
        y = rng.standard_normal(10)
        phi = rng.standard_normal((10, 4))
        weights = rng.standard_normal(4)
        out = ridge(y, phi, weights)
        assert out >= 0

    def test_orth(self) -> None:
        R = np.eye(4)
        assert orth(R) == pytest.approx(0.0)

    def test_div(self, rng: np.random.Generator) -> None:
        a = rng.standard_normal((10, 4))
        b = rng.standard_normal((10, 4))
        out = div(a, b)
        assert -1.0 <= out <= 2.0

    def test_div_zero(self) -> None:
        a = np.zeros((4, 4))
        b = np.zeros((4, 4))
        assert div(a, b) == 0.0

    def test_outer(self, rng: np.random.Generator) -> None:
        y = rng.standard_normal(10)
        phi = rng.standard_normal((10, 4))
        weights = rng.standard_normal(4)
        R = np.eye(4)
        phig = rng.standard_normal((10, 4))
        phil = rng.standard_normal((10, 4))
        out = outer(y, phi, weights, R, phig, phil, wr=0.1, worth=0.0, wdiv=0.1)
        assert isinstance(out, float)


class TestOuterstep:
    def test_step_changes_R(self, outerstep: Outerstep, bundle, rng: np.random.Generator) -> None:
        X_batch = rng.standard_normal((10, 4))
        y_batch = rng.standard_normal(10)

        def feature_fn(R: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            # Return features that depend on R so the gradient is non-zero
            features = X_batch @ R.T
            return features[:, :4], features[:, :4], np.zeros((10, 0))

        new_bundle = outerstep.step(bundle, X_batch, y_batch, feature_fn, rng)
        # R should change (even if by tiny amounts)
        assert not np.array_equal(new_bundle.continuous.R, bundle.continuous.R)


class TestLoop:
    def test_initialize(self, small_loop: Loop, synthetic: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = synthetic
        bundle = small_loop.initialize(X, y)
        assert bundle.weights is not None

    def test_features(self, small_loop: Loop, synthetic: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = synthetic
        bundle = small_loop.initialize(X, y)
        U, phi, phig, phil = small_loop.features(bundle, X)
        assert U.shape[0] == X.shape[0]
        assert phi.shape[1] == phig.shape[1] + phil.shape[1]

    def test_step_no_refresh(self, small_loop: Loop, synthetic: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = synthetic
        bundle = small_loop.initialize(X, y)
        # Without val set, no refresh attempted
        n = X.shape[0]
        n_val = max(1, int(small_loop.plan.val_frac * n))
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]
        bundle = small_loop.step(bundle, X_train, y_train, X_val=X_val, y_val=y_val)
        assert bundle.step == 1


class TestCallbacks:
    def test_log(self, callback_log: Log, bundle) -> None:
        callback_log.onstep(1, bundle)
        callback_log.onstep(2, bundle)

    def test_snapshot(self, callback_snapshot: Snapshot, bundle) -> None:
        callback_snapshot.onrefresh(1, bundle)
        assert len(callback_snapshot.snapshots) == 1

    def test_profile(self, callback_profile: Profile, bundle) -> None:
        callback_profile.onstep(1, bundle)
        callback_profile.oneval(1, {"rmse": 0.1})
        callback_profile.onrefresh(1, bundle)
        assert len(callback_profile.step_times) == 1
        callback_profile.onrefresh(2, bundle)
        assert len(callback_profile.refresh_times) == 1
        assert all(d >= 0.0 for d in callback_profile.refresh_times)
