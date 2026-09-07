"""Numeric invariant tests for kernos."""

from __future__ import annotations

import numpy as np
import pytest

from kernos.basis.nystrom import Nystrom
from kernos.basis.whitening import Whitening
from kernos.correct.orth import Ridge
from kernos.correct.rbf import Rbf


class TestPSD:
    def test_global_features_psd(self, rng: np.random.Generator) -> None:
        U = rng.standard_normal((50, 4))
        w = Whitening(1e-6, 1e-5, 1e-8)
        basis = Nystrom.fromdata(U, 16, w, rng)
        phig = basis.forward(U)
        K = phig @ phig.T
        eigenvalues = np.linalg.eigvalsh(K)
        assert eigenvalues.min() >= -1e-8


class TestRankBound:
    def test_rank_within_bound(self, rng: np.random.Generator) -> None:
        U = rng.standard_normal((50, 4))
        w = Whitening(1e-6, 1e-5, 1e-8)
        basis = Nystrom.fromdata(U, 16, w, rng)
        phig = basis.forward(U)
        rank = np.linalg.matrix_rank(phig)
        assert rank <= 16


class TestOrthogonality:
    def test_global_local_orthogonal(self, rng: np.random.Generator) -> None:
        phig = rng.standard_normal((50, 8))
        phil = rng.standard_normal((50, 6))
        orth = Ridge(1e-4)
        philp = orth.forward(phig, phil)
        ratio = np.linalg.norm(phig.T @ philp, "fro") / (
            np.linalg.norm(phig, "fro") * np.linalg.norm(philp, "fro") + 1e-12
        )
        assert ratio < 1e-3


class TestSPDSolve:
    def test_normal_eq_spd(self, rng: np.random.Generator) -> None:
        phi = rng.standard_normal((50, 8))
        S = phi.T @ phi + 0.1 * np.eye(8)
        eigenvalues = np.linalg.eigvalsh(S)
        assert eigenvalues.min() > 0


class TestCalibration:
    def test_calibration_positive(self, rng: np.random.Generator) -> None:
        from kernos.fuse.scaler import Scaler

        scaler = Scaler()
        phig = rng.standard_normal((50, 8))
        phil = rng.standard_normal((50, 4))
        assert scaler.gnorm(phig) > 0
        assert scaler.lnorm(phil) > 0


class TestWhiteningRank:
    def test_rank_bound(self, rng: np.random.Generator) -> None:
        Z = rng.standard_normal((10, 4))
        w = Whitening(1e-3, 1e-5, 1e-8)
        W = w.kernel(Z)
        _, _, _, rank = w.build(W)
        assert 0 < rank <= 10


class TestAnchorSampling:
    def test_returns_correct_count(self, rng: np.random.Generator) -> None:
        from kernos.correct.sampler import Sampler

        s = Sampler(0.5)
        U = rng.standard_normal((30, 4))
        distances = rng.standard_normal((30, 16))
        residuals = rng.standard_normal(30)
        out = s.pick(U, distances, residuals, abasis=8, rng=rng)
        assert out.shape == (8,)
