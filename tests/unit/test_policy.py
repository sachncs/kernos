"""Unit tests for kernos.policy."""

from __future__ import annotations

import numpy as np
import pytest

from kernos.core.plan import Plan
from kernos.core.state import Bundle
from kernos.policy.budget import Budget
from kernos.policy.drift import Frobenius, Spectral
from kernos.policy.policy import Policy


class TestFrobenius:
    def test_identical_zero(self, drift_frobenius: Frobenius) -> None:
        R = np.eye(4)
        assert drift_frobenius.measure(R, R) == 0.0

    def test_zero_reference(self, drift_frobenius: Frobenius) -> None:
        assert drift_frobenius.measure(np.eye(4), np.zeros((4, 4))) == 0.0

    def test_drift_positive(self, drift_frobenius: Frobenius) -> None:
        assert drift_frobenius.measure(np.eye(4) * 1.1, np.eye(4)) > 0.0


class TestSpectral:
    def test_measure(self, drift_spectral: Spectral) -> None:
        R = np.eye(4)
        out = drift_spectral.measure(R * 1.1, R)
        assert out > 0.0


class TestBudget:
    def test_can(self, budget: Budget) -> None:
        assert budget.can(5.0)
        budget.spend(5.0)
        assert budget.can(4.0)
        assert not budget.can(6.0)

    def test_remaining(self, budget: Budget) -> None:
        budget.spend(3.0)
        assert budget.remaining == pytest.approx(7.0)

    def test_overflow_raises(self, budget: Budget) -> None:
        with pytest.raises(Exception):
            budget.spend(15.0)

    def test_remaining_zero_initially(self, budget: Budget) -> None:
        assert budget.remaining == pytest.approx(10.0)


class TestPolicy:
    def test_decide_all_met(self) -> None:
        plan = Plan(noref=False, drift_hi=0.1, cool=0, warm=0, gain=0.0, rcost=1.0)
        bundle = Bundle(step=10)
        policy = Policy(plan, drift_value=0.5)
        assert policy.decide(bundle, gain=1.0)

    def test_decide_drift_low(self) -> None:
        plan = Plan()
        bundle = Bundle(step=10)
        policy = Policy(plan, drift_value=0.0)
        assert not policy.decide(bundle, gain=1.0)

    def test_decide_noref(self) -> None:
        plan = Plan(noref=True)
        bundle = Bundle(step=100)
        policy = Policy(plan, drift_value=10.0)
        assert not policy.decide(bundle, gain=1.0)

    def test_decide_warmup(self) -> None:
        plan = Plan(warm=10)
        bundle = Bundle(step=5)
        policy = Policy(plan, drift_value=10.0)
        assert not policy.decide(bundle, gain=1.0)
