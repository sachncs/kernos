"""Generated numerical invariant tests for the core linear algebra paths."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from kernos.solver.direct import Direct


@settings(max_examples=20, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=24),
    features=st.integers(min_value=1, max_value=12),
    data=st.data(),
)
def test_direct_solver_is_finite_for_generated_shapes(n: int, features: int, data: st.DataObject) -> None:
    """Direct ridge remains finite for small, including rank-deficient, systems."""
    matrix = data.draw(
        arrays(
            dtype=np.float64,
            shape=(n, features),
            elements=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
        )
    )
    target = data.draw(
        arrays(
            dtype=np.float64,
            shape=(n,),
            elements=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
        )
    )
    solver = Direct(ridge=1e-3, jitter=1e-10, retry=5, factor=10.0, cap=1.0, kappa=1e12)
    weights = solver.solve(matrix, target)
    residual = solver.residual(matrix, target)
    assert weights.shape == (features,)
    assert residual.shape == (n,)
    assert np.isfinite(weights).all()
    assert np.isfinite(residual).all()


@settings(max_examples=20, deadline=None)
@given(
    rows=st.integers(min_value=1, max_value=24),
    cols=st.integers(min_value=1, max_value=12),
)
def test_ridge_solution_is_repeatable_for_generated_shapes(rows: int, cols: int) -> None:
    """Repeated solves on the same generated input are deterministic."""
    rng = np.random.default_rng(rows * 101 + cols)
    matrix = rng.standard_normal((rows, cols))
    target = rng.standard_normal(rows)
    solver = Direct(ridge=1e-3, jitter=1e-10, retry=5, factor=10.0, cap=1.0, kappa=1e12)
    first = solver.solve(matrix, target)
    second = solver.solve(matrix, target)
    np.testing.assert_allclose(first, second, rtol=1e-12, atol=1e-12)
