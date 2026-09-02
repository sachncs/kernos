"""Unit tests for kernos.core.plan."""

from __future__ import annotations

import pytest

from kernos.core.plan import Buffer, Plan


class TestBuffer:
    def test_values(self) -> None:
        assert Buffer.FULL.value == "full"
        assert Buffer.STREAM.value == "stream"
        assert Buffer.ADAPTIVE.value == "adaptive"


class TestPlan:
    def test_defaults(self) -> None:
        plan = Plan()
        assert plan.dim == 64
        assert plan.mbasis == 512
        assert plan.abasis == 128
        assert plan.ridge == pytest.approx(1e-3)
        assert plan.mode == Buffer.FULL

    def test_replace(self) -> None:
        plan = Plan()
        other = plan.replace(dim=32)
        assert other.dim == 32
        assert other.mbasis == plan.mbasis

    def test_validation_abasis_over(self) -> None:
        with pytest.raises(ValueError, match="abasis"):
            Plan(mbasis=16, abasis=8)

    def test_validation_lk_over(self) -> None:
        with pytest.raises(ValueError, match="lk"):
            Plan(lk=10, abasis=4)

    def test_validation_ridge_floor(self) -> None:
        with pytest.raises(ValueError, match="ridge"):
            Plan(ridge=1e-12)

    def test_validation_drift(self) -> None:
        with pytest.raises(ValueError, match="drift_hi"):
            Plan(drift_hi=0.0)

    def test_validation_cool(self) -> None:
        with pytest.raises(ValueError, match="cool"):
            Plan(cool=-1)

    def test_validation_ltau(self) -> None:
        with pytest.raises(ValueError, match="ltau"):
            Plan(ltau=0.0)

    def test_validation_valfrac(self) -> None:
        with pytest.raises(ValueError, match="val_frac"):
            Plan(val_frac=0.0)
        with pytest.raises(ValueError, match="val_frac"):
            Plan(val_frac=1.0)
