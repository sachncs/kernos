"""Unit tests for kernos.correct."""

from __future__ import annotations

import numpy as np
import pytest

from kernos.correct.orth import Ridge, Tikhonov
from kernos.correct.rbf import Rbf
from kernos.correct.sampler import Sampler
from kernos.embed.projector import Projector
from kernos.fuse.scaler import Scaler


class TestRbf:
    def test_forward_shape(self, rng: np.random.Generator) -> None:
        rbf = Rbf(0.1, 4)
        U = rng.standard_normal((20, 4))
        A = rng.standard_normal((8, 4))
        out = rbf.forward(U, A)
        assert out.shape == (20, 8)

    def test_empty_anchor_set_returns_empty_feature_block(self, rng: np.random.Generator) -> None:
        rbf = Rbf(0.1, 4)
        out = rbf.forward(rng.standard_normal((6, 3)), np.zeros((0, 3)))
        assert out.shape == (6, 0)

    def test_rejects_embedding_dimension_mismatch(self, rng: np.random.Generator) -> None:
        rbf = Rbf(0.1, 4)
        with pytest.raises(ValueError, match="embedding dimension"):
            rbf.forward(rng.standard_normal((6, 3)), rng.standard_normal((2, 4)))

    def test_norm_shape(self, rng: np.random.Generator) -> None:
        rbf = Rbf(0.1, 4)
        s = rng.standard_normal((20, 8))
        out = rbf.norm(s)
        assert out.shape == (8,)

    def test_rejects_invalid_configuration(self) -> None:
        with pytest.raises(ValueError, match="ltau"):
            Rbf(0.0, 2)
        with pytest.raises(ValueError, match="lk"):
            Rbf(0.1, 0)


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

    def test_rejects_mismatched_sample_counts(self, rng: np.random.Generator) -> None:
        with pytest.raises(ValueError, match="same sample count"):
            Ridge(1e-4).forward(rng.standard_normal((5, 3)), rng.standard_normal((4, 2)))


class TestTikhonovOrth:
    def test_forward_shape(self, rng: np.random.Generator) -> None:
        orth = Tikhonov(1e-4, n_iter=2)
        phig = rng.standard_normal((50, 8))
        phil = rng.standard_normal((50, 6))
        philp = orth.forward(phig, phil)
        assert philp.shape == phil.shape

    def test_rejects_non_positive_regularization(self) -> None:
        with pytest.raises(ValueError, match="finite and positive"):
            Tikhonov(0.0)


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

    def test_rejects_invalid_mix_and_selection(self, rng: np.random.Generator) -> None:
        with pytest.raises(ValueError, match="amix"):
            Sampler(1.5)
        sampler = Sampler(0.5)
        with pytest.raises(ValueError, match="sample count"):
            sampler.pick(
                rng.standard_normal((3, 2)), rng.standard_normal((3, 4)), rng.standard_normal(3), 4, rng
            )


def test_scaler_rejects_empty_or_non_finite_blocks() -> None:
    scaler = Scaler()
    with pytest.raises(ValueError, match="at least one sample"):
        scaler.gnorm(np.zeros((0, 3)))
    with pytest.raises(ValueError, match="finite"):
        scaler.lnorm(np.array([[np.nan]]))


def test_projector_rejects_dimension_mismatch(rng: np.random.Generator) -> None:
    projector = Projector(np.eye(3))
    with pytest.raises(ValueError, match="input dimension"):
        projector.forward(rng.standard_normal((4, 2)))
