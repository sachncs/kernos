"""Structured runtime diagnostics for estimator training."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any


class SolverEvent(str, Enum):
    """Stable vocabulary for numerical solver recovery and failure events."""

    CHOLESKY_JITTER = "cholesky_jitter"
    PCG_NONCONVERGED = "pcg_nonconverged"
    PCG_BREAKDOWN = "pcg_breakdown"


class SolverStage(str, Enum):
    """Stable stages at which the persistent training solver is invoked."""

    INITIALIZE = "initialize"
    REFRESH = "refresh"


class RecoveryOutcome(str, Enum):
    """Stable outcome vocabulary for a solver invocation."""

    NOT_NEEDED = "not_needed"
    RECOVERED = "recovered"
    NONCONVERGED = "nonconverged"
    FAILED = "failed"


@dataclass(frozen=True)
class SolverRecovery:
    """Immutable record describing one completed or failed solver attempt."""

    stage: SolverStage
    outcome: RecoveryOutcome
    event: SolverEvent | None = None
    attempts: int = 0
    iterations: int = 0
    info: int = 0
    jitter: float | None = None
    condition_number: float | None = None


@dataclass(frozen=True)
class TrainingDiagnostics:
    """Immutable snapshot of observable training progress.

    Attributes:
        steps: Number of completed continuous update steps.
        refreshes: Number of accepted discrete refreshes.
        evaluations: Number of validation evaluations performed.
        samples_seen: Number of accepted training samples processed.
        invalid_batches: Number of rejected training batches.
        invalid_samples: Best-effort sample count from rejected batches.
        last_drift: Most recently measured representation drift.
        last_gain: Most recently measured validation RMSE gain.
        budget_spent: Refresh budget consumed so far.
        estimated_memory_bytes: Conservative preflight estimate for the fit.
        memory_mode: Buffer mode used by the training plan.
        elapsed_seconds: Wall-clock time spent in the current loop.
        solver_iterations: Iterations used by the most recent solver call.
        solver_nonconvergences: Number of solver calls that did not converge.
        last_solver_info: Solver-specific convergence code from the last call.
        solver_fallbacks: Number of calls that needed numerical stabilization.
        last_solver_event: Structured event for the most recent solver call.
        last_solver_jitter: Diagonal jitter used by the most recent factorization.
        last_condition_number: Condition number of the most recent direct solve.
        last_solver_stage: Training stage for the most recent recorded solver call.
        last_solver_recovery: Immutable recovery record for the most recent call.
    """

    steps: int = 0
    refreshes: int = 0
    evaluations: int = 0
    samples_seen: int = 0
    invalid_batches: int = 0
    invalid_samples: int = 0
    last_drift: float | None = None
    last_gain: float | None = None
    budget_spent: float = 0.0
    estimated_memory_bytes: int | None = None
    memory_mode: str | None = None
    elapsed_seconds: float = 0.0
    solver_iterations: int = 0
    solver_nonconvergences: int = 0
    last_solver_info: int | None = None
    solver_fallbacks: int = 0
    last_solver_event: SolverEvent | None = None
    last_solver_jitter: float | None = None
    last_condition_number: float | None = None
    last_solver_stage: SolverStage | None = None
    last_solver_recovery: SolverRecovery | None = None

    def replace(self, **kwargs: Any) -> TrainingDiagnostics:
        """Return a new diagnostics snapshot with selected fields replaced."""
        return replace(self, **kwargs)
