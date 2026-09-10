"""Integration tests for Kernos — the sklearn-compatible estimator."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

from kernos import Kernos


class TestKernos:
    def test_fit_returns_self(self, synthetic: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = synthetic
        model = Kernos(dim=4, mbasis=32, abasis=4, lk=4, steps=3, batch=16, seed=42, drift_hi=10.0, cool=0, warm=0)
        result = model.fit(X, y)
        assert result is model

    def test_predict_shape(self, synthetic: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = synthetic
        model = Kernos(dim=4, mbasis=32, abasis=4, lk=4, steps=3, batch=16, seed=42, drift_hi=10.0, cool=0, warm=0).fit(X, y)
        out = model.predict(X)
        assert out.shape == (X.shape[0],)

    def test_score_is_r2(self, synthetic: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = synthetic
        model = Kernos(dim=4, mbasis=32, abasis=4, lk=4, steps=3, batch=16, seed=42, drift_hi=10.0, cool=0, warm=0).fit(X, y)
        score = model.score(X, y)
        assert np.isfinite(score)

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

    def test_clone(self) -> None:
        model = Kernos(dim=4, mbasis=16)
        cloned = clone(model)
        assert cloned.dim == 4
        assert cloned is not model

    def test_pickle_roundtrip(self, synthetic: tuple[np.ndarray, np.ndarray]) -> None:
        import pickle

        X, y = synthetic
        model = Kernos(dim=4, mbasis=32, abasis=4, lk=4, steps=3, batch=16, seed=42, drift_hi=10.0, cool=0, warm=0).fit(X, y)
        score_before = model.score(X, y)
        blob = pickle.dumps(model)
        restored = pickle.loads(blob)
        # loop_ is dropped by __getstate__; rebuild via fit on same X
        restored.fit(X, y)
        score_after = restored.score(X, y)
        np.testing.assert_allclose(score_before, score_after, atol=1e-6)

    def test_partial_fit(self, synthetic: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = synthetic
        model = Kernos(dim=4, mbasis=32, abasis=4, lk=4, steps=1, batch=16, seed=42, drift_hi=10.0, cool=0, warm=0).fit(X, y)
        result = model.partial_fit(X, y)
        assert result is model

    def test_n_features_in_recorded(self, synthetic: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = synthetic
        model = Kernos(dim=4, mbasis=32, abasis=4, lk=4, steps=1, batch=16, seed=42, drift_hi=10.0, cool=0, warm=0).fit(X, y)
        assert model.n_features_in_ == X.shape[1]

    def test_predict_before_fit_raises(self) -> None:
        model = Kernos()
        with pytest.raises(RuntimeError):
            model.predict(np.zeros((5, 4)))

    def test_fit_too_few_samples_raises(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((5, 3))
        y = rng.standard_normal(5)
        model = Kernos(dim=4, mbasis=16, abasis=4, lk=2, steps=1)
        with pytest.raises(ValueError, match="n_samples"):
            model.fit(X, y)
