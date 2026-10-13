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

    def test_grouped_configuration_views_are_immutable(self) -> None:
        plan = Plan(dim=8, mbasis=16, abasis=4, lk=4)
        assert plan.embedding_config.dim == 8
        assert plan.basis_config.mbasis == 16
        assert plan.solver_config.ridge == pytest.approx(plan.ridge)
        assert plan.solver_config.stab_lmin == pytest.approx(plan.stab_lmin)
        assert plan.training_config.mode is Buffer.FULL
        assert plan.refresh_config.rcost == pytest.approx(plan.rcost)
        assert plan.ablation_config.noref is False
        assert plan.validation_config.val_frac == pytest.approx(plan.val_frac)
        with pytest.raises(AttributeError):
            plan.training_config.steps = 2  # type: ignore[misc]

    def test_descriptive_aliases_preserve_legacy_values(self) -> None:
        plan = Plan(wr=0.2, worth=0.3, wdiv=0.4, fdeps=1e-4, cool=3, warm=4, gain=0.5, rcost=0.6)
        assert plan.representation_regularization == pytest.approx(0.2)
        assert plan.orthogonality_regularization == pytest.approx(0.3)
        assert plan.diversity_regularization == pytest.approx(0.4)
        assert plan.finite_difference_step == pytest.approx(1e-4)
        assert plan.refresh_cooldown == 3
        assert plan.refresh_warmup == 4
        assert plan.refresh_gain_threshold == pytest.approx(0.5)
        assert plan.refresh_cost == pytest.approx(0.6)

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

    def test_validation_stability_controls(self) -> None:
        with pytest.raises(ValueError, match="stab_tau"):
            Plan(stab_tau=0.0)
        with pytest.raises(ValueError, match="stab_jitter_max"):
            Plan(stab_jitter=1.0, stab_jitter_max=0.5)

    def test_memory_estimate_and_validation(self) -> None:
        plan = Plan(dim=4, mbasis=16, abasis=4, lk=4, memory_limit_mb=10.0)
        assert plan.estimate_memory_bytes(10, 4) > 0
        with pytest.raises(ValueError, match="memory_limit_mb"):
            Plan(memory_limit_mb=0.0)
