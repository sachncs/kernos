"""Unit tests for kernos.core.state."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from kernos.core.state import Bundle, Continuous, Discrete


class TestContinuous:
    def test_defaults(self) -> None:
        c = Continuous()
        assert c.theta is None
        assert c.R is None

    def test_immutable(self) -> None:
        c = Continuous()
        with pytest.raises(FrozenInstanceError):
            c.R = np.eye(4)  # type: ignore[misc]

    def test_projection_array_is_owned_and_read_only(self) -> None:
        source = np.eye(4)
        c = Continuous(R=source)
        source[0, 0] = 99.0
        assert c.R is not None
        assert c.R[0, 0] == 1.0
        with pytest.raises(ValueError):
            c.R[0, 0] = 2.0


class TestDiscrete:
    def test_defaults(self) -> None:
        d = Discrete()
        assert d.landmarks is None
        assert d.cglobal == 1.0
        assert d.clocal == 1.0
        assert d.active == 1
        assert d.gateval == 0.5
        assert d.tlast == 0


class TestBundle:
    def test_defaults(self) -> None:
        b = Bundle()
        assert b.step == 0
        assert b.weights is None

    def test_replace(self) -> None:
        b = Bundle()
        other = b.replace(step=5)
        assert other.step == 5
        assert b.step == 0

    def test_weights_are_read_only(self) -> None:
        b = Bundle(weights=np.ones(3))
        assert b.weights is not None
        with pytest.raises(ValueError):
            b.weights[0] = 2.0
