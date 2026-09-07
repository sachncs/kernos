"""Unit tests for kernos.core.types."""

from __future__ import annotations

import numpy as np
import pytest

from kernos.core.types import Array, asfloat64, check


class TestCheck:
    def test_passes_when_true(self) -> None:
        check(True, "should not raise")

    def test_raises_when_false(self) -> None:
        with pytest.raises(ValueError, match="boom"):
            check(False, "boom")


class TestAsfloat64:
    def test_passthrough(self, rng: np.random.Generator) -> None:
        x = rng.standard_normal(10)
        assert asfloat64(x) is x

    def test_cast(self, rng: np.random.Generator) -> None:
        x = rng.standard_normal(10).astype(np.float32)
        out = asfloat64(x)
        assert out.dtype == np.float64
