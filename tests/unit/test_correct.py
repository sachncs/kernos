"""Unit tests for kernos.correct."""

from __future__ import annotations

import numpy as np
import pytest

from kernos.correct.orth import Ridge, Tikhonov
from kernos.correct.rbf import Rbf
from kernos.correct.sampler import Sampler


class TestRbf:
    def test_forward_shape(self, rng: np.random.Generator) -> None:
        rbf = Rbf(0.1, 4)
        U = rng.standard_normal((20, 4))
        A = rng.standard_normal((8, 4))
        out = rbf.forward(U, A)
        assert out.shape == (20, 8)

    def test_norm_shape(self, rng: np.random.Generator) -> None:
        rbf = Rbf(0.1, 4)
        s = rng.standard_normal((20, 8))
        out = rbf.norm(s)
        assert out.shape == (8,)


class TestRidgeOrth:
    def test_project_shape(self, rng: np.random.Generator) -> None:
        orth = Ridge(1e-4)
        phig = rng.standard_normal((20, 8))
        out = orth.project(phig)
        # project returns the thin operator (rank, samples)
        assert out.shape == (8, 20)

    def test_forward_orthogonal(self, rng: np.random.Generator) -> None:
        orth = Ridge(1e-4)
        phig = rng.standard_normal((50, 8))
        phil = rng.standard_normal((50, 6))
        philp = orth.forward(phig, phil)
        assert philp.shape == phil.shape
        assert orth.check(phig, philp, tol=1e-3)


class TestTikhonovOrth:
    def test_forward_shape(self, rng: np.random.Generator) -> None:
        orth = Tikhonov(1e-4, n_iter=2)
        phig = rng.standard_normal((50, 8))
        phil = rng.standard_normal((50, 6))
        philp = orth.forward(phig, phil)
        assert philp.shape == phil.shape


class TestSampler:
    def test_cover(self, rng: np.random.Generator) -> None:
        s = Sampler(0.5)
        distances = rng.standard_normal((10, 8))
        out = s.cover(distances)
        assert out.shape == (10,)

    def test_resid(self, rng: np.random.Generator) -> None:
        s = Sampler(0.5)
        residuals = rng.standard_normal(10)
        out = s.resid(residuals)
        assert out.shape == (10,)
        assert (out >= 0).all()

    def test_pick(self, rng: np.random.Generator) -> None:
        s = Sampler(0.5)
        U = rng.standard_normal((20, 4))
        distances = rng.standard_normal((20, 8))
        residuals = rng.standard_normal(20)
        out = s.pick(U, distances, residuals, abasis=5, rng=rng)
        assert out.shape == (5,)
