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
    def test_precon(self, rng: np.random.Generator) -> None:
        S = rng.standard_normal((4, 4))
        S = S @ S.T + np.eye(4)
        j = Jacobi()
        out = j.precon(S)
        np.testing.assert_allclose(out, 1.0 / np.maximum(np.diag(S), 1e-12))


class TestDirect:
    def test_solve(self, direct_solver: Direct, rng: np.random.Generator) -> None:
        phi = rng.standard_normal((50, 8))
        y = rng.standard_normal(50)
        w = direct_solver.solve(phi, y)
        assert w.shape == (8,)

    def test_residual(self, direct_solver: Direct, rng: np.random.Generator) -> None:
        phi = rng.standard_normal((50, 8))
        y = rng.standard_normal(50)
        out = direct_solver.residual(phi, y)
        assert out.shape == (50,)


class TestIterative:
    def test_solve(self, iterative_solver: Iterative, rng: np.random.Generator) -> None:
        phi = rng.standard_normal((50, 8))
        y = rng.standard_normal(50)
        w = iterative_solver.solve(phi, y)
        assert w.shape == (8,)

    def test_precon(self, iterative_solver: Iterative, rng: np.random.Generator) -> None:
        S = rng.standard_normal((8, 8))
        S = S @ S.T + np.eye(8)
        out = iterative_solver.precon(S)
        assert out.shape == (8,)


class TestWoodbury:
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
