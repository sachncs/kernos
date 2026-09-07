"""Unit tests for kernos.bench."""

from __future__ import annotations

import numpy as np
import pytest

from kernos.bench.baseline import Nystrom, Random, Ridge
from kernos.bench.dataset import hetero, highdim, linear, poly, split
from kernos.bench.metric import allmetrics, mae, maxerr, r2, rmse
from kernos.bench.runner import Report, Runner


class TestMetric:
    def test_rmse(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        yhat = np.array([1.0, 2.0, 4.0])
        assert rmse(y, yhat) == pytest.approx(np.sqrt(1.0 / 3.0))

    def test_r2_constant_returns_nan(self) -> None:
        y = np.ones(5)
        assert np.isnan(r2(y, y))

    def test_r2_perfect(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert r2(y, y) == pytest.approx(1.0)

    def test_mae(self) -> None:
        y = np.array([1.0, 2.0])
        yhat = np.array([3.0, 0.0])
        assert mae(y, yhat) == pytest.approx(2.0)

    def test_maxerr(self) -> None:
        y = np.array([1.0, 2.0])
        yhat = np.array([4.0, 2.0])
        assert maxerr(y, yhat) == pytest.approx(3.0)

    def test_allmetrics(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        yhat = np.array([1.0, 2.0, 4.0])
        out = allmetrics(y, yhat)
        assert "rmse" in out
        assert "r2" in out


class TestDataset:
    def test_linear(self, rng: np.random.Generator) -> None:
        X, y = linear(rng, 50, 3)
        assert X.shape == (50, 3)
        assert y.shape == (50,)

    def test_poly(self, rng: np.random.Generator) -> None:
        X, y = poly(rng, 50, degree=3)
        assert X.shape == (50, 1)
        assert y.shape == (50,)

    def test_highdim(self, rng: np.random.Generator) -> None:
        X, y = highdim(rng, 50, features=10, informative=3)
        assert X.shape == (50, 10)

    def test_hetero(self, rng: np.random.Generator) -> None:
        X, y = hetero(rng, 50)
        assert X.shape == (50, 1)

    def test_split(self, rng: np.random.Generator) -> None:
        X = rng.standard_normal((100, 4))
        y = rng.standard_normal(100)
        Xtr, Xte, ytr, yte = split(X, y, 0.2, rng)
        assert Xtr.shape[0] == 80
        assert Xte.shape[0] == 20


class TestBaselines:
    def test_ridge(self, synthetic: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = synthetic
        m = Ridge(ridge=1e-2)
        m.fit(X, y)
        yhat = m.predict(X)
        assert yhat.shape == y.shape

    def test_nystrom(self, synthetic: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = synthetic
        m = Nystrom(mbasis=16, ridge=1e-2, seed=42)
        m.fit(X, y)
        yhat = m.predict(X)
        assert yhat.shape == y.shape

    def test_random(self, synthetic: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = synthetic
        m = Random(mfeat=64, ridge=1e-2, seed=42)
        m.fit(X, y)
        yhat = m.predict(X)
        assert yhat.shape == y.shape


class TestRunner:
    def test_run(self, synthetic: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = synthetic
        Xtr, Xte, ytr, yte = split(X, y, 0.2, np.random.default_rng(0))
        runner = Runner()
        r = runner.run("ridge", Ridge(ridge=1e-2), Xtr, ytr, Xte, yte)
        assert r.name == "ridge"
        assert r.rmse >= 0

    def test_suite(self, synthetic: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = synthetic
        Xtr, Xte, ytr, yte = split(X, y, 0.2, np.random.default_rng(0))
        runner = Runner()
        out = runner.suite({"a": Ridge(), "b": Ridge()}, Xtr, ytr, Xte, yte)
        assert len(out) == 2

    def test_format(self) -> None:
        runner = Runner()
        out = runner.format([Report(name="a", rmse=0.1, extra={})])
        assert "| a |" in out
