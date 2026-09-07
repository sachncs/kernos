"""Tests for kernos.sample."""

from __future__ import annotations

import numpy as np
import pytest

from kernos.sample import kmeanspp


class TestKmeanspp:
    def test_shape(self, rng: np.random.Generator) -> None:
        X = rng.standard_normal((50, 4))
        out = kmeanspp(X, k=8, rng=rng)
        assert out.shape == (8,)

    def test_unique(self, rng: np.random.Generator) -> None:
        X = rng.standard_normal((50, 4))
        out = kmeanspp(X, k=10, rng=rng)
        assert len(set(out.tolist())) == 10

    def test_rejects_k_too_large(self, rng: np.random.Generator) -> None:
        X = rng.standard_normal((5, 2))
        with pytest.raises(ValueError, match="k"):
            kmeanspp(X, k=10, rng=rng)

    def test_uniform_when_duplicates(self, rng: np.random.Generator) -> None:
        X = np.zeros((20, 4))  # all identical
        out = kmeanspp(X, k=5, rng=rng)
        assert out.shape == (5,)
