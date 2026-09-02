"""Unit tests for kernos.embed."""

from __future__ import annotations

import numpy as np
import pytest

from kernos.embed.identity import Identity
from kernos.embed.kernel import Kernel
from kernos.embed.linear import Linear
from kernos.embed.projector import Projector, norm


class TestLinear:
    def test_forward_shape(self, rng: np.random.Generator) -> None:
        e = Linear(4, 8, rng)
        x = rng.standard_normal((10, 4))
        out = e.forward(x)
        assert out.shape == (10, 8)

    def test_params_roundtrip(self, rng: np.random.Generator) -> None:
        e = Linear(4, 4, rng)
        weights, bias = e.params()
        e.setparams(weights * 2.0, bias + 1.0)
        out = e.forward(np.eye(4))
        assert out.shape == (4, 4)


class TestKernel:
    def test_forward_shape(self, rng: np.random.Generator) -> None:
        centers = rng.standard_normal((5, 4))
        e = Kernel(centers, gamma=1.0)
        x = rng.standard_normal((10, 4))
        out = e.forward(x)
        assert out.shape == (10, 5)


class TestIdentity:
    def test_forward_passthrough(self) -> None:
        e = Identity()
        x = np.arange(12).reshape(3, 4).astype(np.float64)
        out = e.forward(x)
        np.testing.assert_array_equal(out, x)


class TestNorm:
    def test_unit_rows(self) -> None:
        x = np.arange(20, dtype=np.float64).reshape(5, 4)
        out = norm(x)
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, np.ones(5), atol=1e-6)

    def test_zero_safe(self) -> None:
        x = np.zeros((3, 4))
        out = norm(x)
        assert out.shape == (3, 4)
        assert np.all(np.isfinite(out))


class TestProjector:
    def test_forward_shape(self) -> None:
        R = np.eye(4)
        p = Projector(R)
        u = np.arange(12, dtype=np.float64).reshape(3, 4)
        out = p.forward(u)
        assert out.shape == (3, 4)
