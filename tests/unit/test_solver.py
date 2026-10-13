"""Unit tests for kernos.solver."""

from __future__ import annotations

import numpy as np
import pytest

from kernos.core.error import IllConditionedError
from kernos.solver.direct import Direct
from kernos.solver.equation import assemble, crossvec, gramian
from kernos.solver.iterative import Iterative
from kernos.solver.jacobi import Jacobi
from kernos.solver.woodbury import Woodbury


class TestEquation:
    def test_gramian(self, rng: np.random.Generator) -> None:
        phi = rng.standard_normal((20, 4))
        out = gramian(phi)
        assert out.shape == (4, 4)
        np.testing.assert_allclose(out, out.T, atol=1e-12)

    def test_crossvec(self, rng: np.random.Generator) -> None:
        phi = rng.standard_normal((20, 4))
        y = rng.standard_normal(20)
        out = crossvec(phi, y)
        assert out.shape == (4,)

    def test_assemble(self, rng: np.random.Generator) -> None:
        phi = rng.standard_normal((20, 4))
        y = rng.standard_normal(20)
        S, b = assemble(phi, y, ridge=0.1)
        assert S.shape == (4, 4)
        assert b.shape == (4,)


class TestJacobi:
    def test_rejects_invalid_epsilon(self) -> None:
        with pytest.raises(ValueError, match="eps"):
            Jacobi(eps=0.0)

    def test_precon(self, rng: np.random.Generator) -> None:
        S = rng.standard_normal((4, 4))
        S = S @ S.T + np.eye(4)
        j = Jacobi()
        out = j.precon(S)
        np.testing.assert_allclose(out, 1.0 / np.maximum(np.diag(S), 1e-12))


class TestDirect:
    def test_rejects_invalid_stability_configuration(self) -> None:
        with pytest.raises(ValueError, match="ridge"):
            Direct(0.0, 1e-10, 5, 10.0, 1.0, 1e12)
        with pytest.raises(ValueError, match="factor"):
            Direct(1e-3, 1e-10, 5, 1.0, 1.0, 1e12)

    def test_solve(self, direct_solver: Direct, rng: np.random.Generator) -> None:
        phi = rng.standard_normal((50, 8))
        y = rng.standard_normal(50)
        w = direct_solver.solve(phi, y)
        assert w.shape == (8,)
        assert direct_solver.last_jitter is None
        assert direct_solver.last_condition is not None

    def test_rejects_conditioning_above_configured_limit(self, rng: np.random.Generator) -> None:
        solver = Direct(1e-3, 1e-10, 5, 10.0, 1.0, kappa=1.0)
        with pytest.raises(IllConditionedError, match="normal-equation matrix condition"):
            solver.solve(rng.standard_normal((12, 3)), rng.standard_normal(12))

    def test_residual(self, direct_solver: Direct, rng: np.random.Generator) -> None:
        phi = rng.standard_normal((50, 8))
        y = rng.standard_normal(50)
        out = direct_solver.residual(phi, y)
        assert out.shape == (50,)

    def test_rejects_non_finite_system(self, direct_solver: Direct) -> None:
        phi = np.ones((4, 2))
        y = np.ones(4)
        phi[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            direct_solver.solve(phi, y)

    def test_rejects_mismatched_system(self, direct_solver: Direct) -> None:
        with pytest.raises(ValueError, match="disagree"):
            direct_solver.solve(np.ones((4, 2)), np.ones(3))


class TestIterative:
    def test_rejects_invalid_configuration(self) -> None:
        with pytest.raises(ValueError, match="max_iter"):
            Iterative(0.1, max_iter=0)
        with pytest.raises(ValueError, match="tol"):
            Iterative(0.1, tol=0.0)

    def test_solve(self, iterative_solver: Iterative, rng: np.random.Generator) -> None:
        phi = rng.standard_normal((50, 8))
        y = rng.standard_normal(50)
        w = iterative_solver.solve(phi, y)
        assert w.shape == (8,)
        assert iterative_solver.last_iterations < iterative_solver.max_iter

    def test_precon(self, iterative_solver: Iterative, rng: np.random.Generator) -> None:
        S = rng.standard_normal((8, 8))
        S = S @ S.T + np.eye(8)
        out = iterative_solver.precon(S)
        assert out.shape == (8,)

    def test_records_non_convergence(
        self, rng: np.random.Generator, caplog: pytest.LogCaptureFixture
    ) -> None:
        solver = Iterative(ridge=0.1, max_iter=1, tol=1e-30)
        phi = rng.standard_normal((50, 8))
        y = rng.standard_normal(50)
        with caplog.at_level("WARNING", logger="kernos.solver"):
            solver.solve(phi, y)
        assert solver.last_info > 0
        assert solver.last_iterations == 1
        assert solver.last_solver_event == "pcg_nonconverged"
        assert "did not converge" in caplog.text

    def test_breakdown_raises_typed_error(self) -> None:
        solver = Iterative(ridge=0.1)
        with pytest.raises(IllConditionedError, match="broke down"):
            solver.solve(np.full((2, 2), 1e200), np.ones(2))


class TestWoodbury:
    def test_rejects_invalid_configuration(self) -> None:
        with pytest.raises(ValueError, match="cap"):
            Woodbury(0.1, 1.0, 2, 10.0, 0.5)

    def test_solve(self, woodbury_solver: Woodbury, rng: np.random.Generator) -> None:
        phi = rng.standard_normal((5, 10))
        y = rng.standard_normal(5)
        w = woodbury_solver.solve(phi, y)
        assert w.shape == (10,)

    def test_rejects_overdetermined(self, woodbury_solver: Woodbury, rng: np.random.Generator) -> None:
        phi = rng.standard_normal((20, 4))
        y = rng.standard_normal(20)
        with pytest.raises(IllConditionedError):
            woodbury_solver.solve(phi, y)
