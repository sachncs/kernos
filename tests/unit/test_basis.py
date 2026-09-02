"""Unit tests for kernos.basis."""

from __future__ import annotations

import numpy as np
import pytest

from kernos.basis.greedy import Greedy
from kernos.basis.nystrom import Nystrom
from kernos.basis.random import Random
from kernos.basis.whitening import Whitening
from kernos.core.error import ShapeError


class TestWhitening:
    def test_kernel_shape(self, rng: np.random.Generator) -> None:
        Z = rng.standard_normal((8, 4))
        w = Whitening(1e-6, 1e-5, 1e-8)
        W = w.kernel(Z)
        assert W.shape == (8, 8)
        np.testing.assert_allclose(W, W.T, atol=1e-12)

    def test_build_rank(self, rng: np.random.Generator) -> None:
        Z = rng.standard_normal((16, 4))
        w = Whitening(1e-3, 1e-5, 1e-8)
        W = w.kernel(Z)
        _, ev, Mg, rank = w.build(W)
        assert Mg.shape == (16, rank)
        assert rank > 0

    def test_degenerate_raises(self) -> None:
        w = Whitening(1e6, 1e-5, 1e-8)
        with pytest.raises(ShapeError):
            w.build(np.eye(4))


class TestNystrom:
    def test_fromdata(self, rng: np.random.Generator, whitening: Whitening) -> None:
        U = rng.standard_normal((40, 4))
        basis = Nystrom.fromdata(U, 16, whitening, rng)
        out = basis.forward(U)
        assert out.shape[0] == 40
        assert out.shape[1] > 0

    def test_forward_shape(self, rng: np.random.Generator, whitening: Whitening) -> None:
        U = rng.standard_normal((40, 4))
        basis = Nystrom.fromdata(U, 16, whitening, rng)
        q = rng.standard_normal((10, 4))
        out = basis.forward(q)
        assert out.shape[0] == 10

    def test_fromlandmarks(self, rng: np.random.Generator, whitening: Whitening) -> None:
        Z = rng.standard_normal((16, 4))
        basis = Nystrom.fromlandmarks(Z, whitening)
        q = rng.standard_normal((10, 4))
        out = basis.forward(q)
        assert out.shape[0] == 10


class TestRandom:
    def test_fromdata(self, rng: np.random.Generator) -> None:
        basis = Random.fromdata(4, 64, rng, gamma=1.0)
        assert basis.omega.shape == (64, 4)
        assert basis.phases.shape == (64,)

    def test_forward_shape(self, rng: np.random.Generator) -> None:
        basis = Random.fromdata(4, 64, rng)
        x = rng.standard_normal((10, 4))
        out = basis.forward(x)
        assert out.shape == (10, 64)


class TestGreedy:
    def test_fromdata(self, rng: np.random.Generator) -> None:
        basis = Greedy.fromdata(rng.standard_normal((40, 4)), 16, rng)
        assert basis.landmarks.shape == (16, 4)
        out = basis.forward(rng.standard_normal((10, 4)))
        assert out.shape == (10, 16)
