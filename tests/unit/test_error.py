"""Unit tests for kernos.core.error."""

from __future__ import annotations

import pytest

from kernos.core.error import AwarenessError, BudgetExceeded, IllConditionedError, ShapeError


class TestHierarchy:
    def test_ill_conditioned(self) -> None:
        assert issubclass(IllConditionedError, AwarenessError)
        with pytest.raises(AwarenessError):
            raise IllConditionedError("cond too high")

    def test_budget_exceeded(self) -> None:
        with pytest.raises(AwarenessError):
            raise BudgetExceeded("over budget")

    def test_shape_error(self) -> None:
        assert issubclass(ShapeError, AwarenessError)
        assert issubclass(ShapeError, ValueError)
        with pytest.raises(ValueError):
            raise ShapeError("shape mismatch")
