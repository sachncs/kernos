"""Unit tests for kernos.core.error."""

from __future__ import annotations

import pytest

from kernos.core.diagnostics import RecoveryOutcome, SolverEvent, SolverRecovery, SolverStage
from kernos.core.error import (
    AwarenessError,
    BudgetExceeded,
    IllConditionedError,
    MemoryLimitExceeded,
    ShapeError,
    TrainingCancelled,
    TrainingTimeout,
)


class TestHierarchy:
    def test_solver_event_is_string_compatible_and_stable(self) -> None:
        assert SolverEvent.PCG_BREAKDOWN == "pcg_breakdown"
        assert {event.value for event in SolverEvent} == {
            "cholesky_jitter",
            "pcg_nonconverged",
            "pcg_breakdown",
        }

    def test_solver_stage_is_string_compatible_and_stable(self) -> None:
        assert SolverStage.INITIALIZE == "initialize"
        assert {stage.value for stage in SolverStage} == {"initialize", "refresh"}

    def test_recovery_outcome_is_string_compatible_and_stable(self) -> None:
        assert RecoveryOutcome.RECOVERED == "recovered"
        assert {outcome.value for outcome in RecoveryOutcome} == {
            "not_needed",
            "recovered",
            "nonconverged",
            "failed",
        }

    def test_solver_recovery_is_immutable(self) -> None:
        record = SolverRecovery(SolverStage.INITIALIZE, RecoveryOutcome.NOT_NEEDED)
        with pytest.raises(AttributeError):
            record.outcome = RecoveryOutcome.FAILED  # type: ignore[misc]

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

    def test_memory_limit_exceeded(self) -> None:
        assert issubclass(MemoryLimitExceeded, AwarenessError)
        assert issubclass(MemoryLimitExceeded, MemoryError)
        with pytest.raises(MemoryError):
            raise MemoryLimitExceeded("memory budget")

    def test_training_cancelled(self) -> None:
        assert issubclass(TrainingCancelled, AwarenessError)
        with pytest.raises(AwarenessError):
            raise TrainingCancelled("stop")

    def test_training_timeout(self) -> None:
        assert issubclass(TrainingTimeout, TrainingCancelled)
        with pytest.raises(AwarenessError):
            raise TrainingTimeout("timeout")
