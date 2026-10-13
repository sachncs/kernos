"""Numeric tests for kernos.numeric and kernos.linalg."""

from __future__ import annotations

import numpy as np
import pytest

from kernos.core.error import IllConditionedError
from kernos.numeric import (
    chol,
    chol_with_jitter,
    clipeig,
    condition,
    drift,
    epsilon,
    fnorm,
    pcg,
    pcg_detailed,
    retain,
    truncate,
)


class TestClipeig:
    def test_floor(self) -> None:
        evals = np.array([-1.0, 0.0, 1.0])
        out = clipeig(evals, 0.0)
        assert (out >= 0.0).all()


class TestTruncate:
    def test_soft(self) -> None:
        evals = np.array([0.0, 1.0, 2.0])
        out = truncate(evals, tau=0.1, eps=0.1)
        assert (out >= 0.0).all()


class TestEpsilon:
    def test_value(self) -> None:
        out = epsilon(tr=10.0, mbasis=4, alpha=1e-5)
        assert out == pytest.approx(10.0 * 1e-5 / 4)


class TestRetain:
    def test_indices(self) -> None:
        evals = np.array([0.0, 1e-3, 0.1, 1.0])
        indices, rank = retain(evals, tau=1e-2)
        assert rank == 2


def test_pcg_detailed_reports_convergence_iteration() -> None:
    """The detailed solver path reports work actually performed."""
    S = np.eye(3)
    rhs = np.ones(3)
    solution, info, iterations = pcg_detailed(S, rhs, max_iter=50, tol=1e-12)
    assert info == 0
    assert iterations == 1
    np.testing.assert_allclose(S @ solution, rhs, atol=1e-12)


class TestCondition:
    def test_rejects_invalid_matrix_and_threshold(self) -> None:
        with pytest.raises(ValueError, match="square"):
            condition(np.ones((2, 3)), kappa=10.0)
        with pytest.raises(ValueError, match="kappa"):
            condition(np.eye(2), kappa=0.0)

    def test_under_threshold(self) -> None:
        A = np.eye(4)
        assert condition(A, kappa=1e10) > 0.0

    def test_over_threshold_raises(self) -> None:
        A = np.diag([1.0, 1.0, 1.0, 1e-15])
        with pytest.raises(IllConditionedError):
            condition(A, kappa=1e6)


class TestChol:
    def test_rejects_invalid_controls(self) -> None:
        with pytest.raises(ValueError, match="retry"):
            chol_with_jitter(np.eye(2), retry=-1)
        with pytest.raises(ValueError, match="cap"):
            chol_with_jitter(np.eye(2), jitter=1.0, cap=0.5)

    def test_basic(self) -> None:
        A = np.eye(4) + 0.1 * np.ones((4, 4))
        L = chol(A, jitter=1e-10, retry=5)
        np.testing.assert_allclose(L @ L.T, A, atol=1e-6)

    def test_jitter_fallback(self) -> None:
        A = np.array([[0.0, 0.0], [0.0, 0.0]])  # singular PSD
        # Should succeed because chol adds jitter; the fallback handles singular input.
        L = chol(A, jitter=1e-3, retry=3)
        assert L.shape == (2, 2)

    def test_reports_fallback_metadata(self) -> None:
        factor, used_jitter, attempts = chol_with_jitter(np.diag([-5e-4, 1.0]), jitter=1e-4, retry=2)
        assert factor.shape == (2, 2)
        assert used_jitter == pytest.approx(1e-3)
        assert attempts == 1


class TestFnorm:
    def test_value(self) -> None:
        A = np.eye(4)
        assert fnorm(A) == pytest.approx(2.0)


class TestDrift:
    def test_zero_ref(self) -> None:
        assert drift(np.eye(4), np.zeros((4, 4))) == 0.0

    def test_identical(self) -> None:
        assert drift(np.eye(4), np.eye(4)) == 0.0

    def test_drift(self) -> None:
        assert drift(np.eye(4) * 1.1, np.eye(4)) > 0.0


class TestPcg:
    def test_rejects_invalid_system(self) -> None:
        with pytest.raises(ValueError, match="symmetric"):
            pcg(np.array([[1.0, 2.0], [0.0, 1.0]]), np.ones(2))
        with pytest.raises(ValueError, match="max_iter"):
            pcg(np.eye(2), np.ones(2), max_iter=0)

    def test_solves_identity(self) -> None:
        b = np.array([1.0, 2.0, 3.0])
        x, info = pcg(np.eye(3), b, tol=1e-10)
        np.testing.assert_allclose(x, b, atol=1e-6)
        assert info == 0

    def test_reports_breakdown_for_non_positive_curvature(self) -> None:
        x, info = pcg(np.diag([-1.0, 1.0]), np.ones(2))
        assert info < 0
        assert np.isfinite(x).all()

    def test_detailed_reports_nonconvergence_and_breakdown_iterations(self) -> None:
        _, info, iterations = pcg_detailed(np.diag([1.0, 2.0]), np.ones(2), max_iter=1, tol=1e-30)
        assert info == 1
        assert iterations == 1

        _, info, iterations = pcg_detailed(np.diag([-1.0, 1.0]), np.ones(2))
        assert info < 0
        assert iterations == abs(info)
