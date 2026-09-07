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
