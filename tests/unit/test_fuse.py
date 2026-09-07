"""Unit tests for kernos.fuse."""

from __future__ import annotations

import numpy as np
import pytest

from kernos.fuse.fuse import Fuse
from kernos.fuse.gate import Gate
from kernos.fuse.scaler import Scaler


class TestGate:
    def test_sigmoid(self) -> None:
        g = Gate()
        assert g.sigmoid(0.0) == pytest.approx(0.5)
        assert g.sigmoid(10.0) > 0.999
        assert g.sigmoid(-10.0) < 0.001

    def test_value_freeze(self) -> None:
        g = Gate(gatelogit=0.0, nofreeze=False)
        _ = g.value(1.0)
        assert g.gatelogit == 1.0
        g_freeze = Gate(gatelogit=0.0, nofreeze=True)
        _ = g_freeze.value(1.0)
        assert g_freeze.gatelogit == 0.0


class TestScaler:
    def test_gnorm(self, rng: np.random.Generator) -> None:
        scaler = Scaler()
        phig = rng.standard_normal((100, 8))
        out = scaler.gnorm(phig)
        assert out > 0

    def test_lnorm(self, rng: np.random.Generator) -> None:
        scaler = Scaler()
        phil = rng.standard_normal((100, 4))
        out = scaler.lnorm(phil)
        assert out > 0

    def test_scale(self) -> None:
        scaler = Scaler()
        x = np.array([[2.0, 4.0]])
        out = scaler.scale(x, 2.0)
        np.testing.assert_allclose(out, [[1.0, 2.0]])


class TestFuse:
    def test_forward_shape(self, rng: np.random.Generator) -> None:
        fuse = Fuse()
        phig = rng.standard_normal((10, 8))
        phil = rng.standard_normal((10, 4))
        out = fuse.forward(phig, phil)
        assert out.shape == (10, 12)

    def test_split(self, rng: np.random.Generator) -> None:
        fuse = Fuse()
        phig = rng.standard_normal((10, 8))
        phil = rng.standard_normal((10, 4))
        phi = fuse.forward(phig, phil)
        g, l = fuse.split(phi, 8)
        np.testing.assert_allclose(g, phig * np.sqrt(0.5), atol=1e-8)
