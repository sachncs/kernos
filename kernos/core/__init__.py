"""Core module: foundational types, plans, state, errors, RNG."""

from kernos.core.config import (
    AblationConfig,
    BasisConfig,
    EmbeddingConfig,
    RefreshConfig,
    SolverConfig,
    TrainingConfig,
    ValidationConfig,
)
from kernos.core.diagnostics import (
    RecoveryOutcome,
    SolverEvent,
    SolverRecovery,
    SolverStage,
    TrainingDiagnostics,
)
from kernos.core.error import (
    AwarenessError,
    BudgetExceeded,
    IllConditionedError,
    MemoryLimitExceeded,
    ShapeError,
    TrainingCancelled,
    TrainingTimeout,
)
from kernos.core.factory import build_drift_metric, build_embedder, build_solver
from kernos.core.interfaces import (
    AnchorSampler,
    Basis,
    BudgetAccountant,
    DriftMetric,
    FeatureFusion,
    FeatureMap,
    LocalFeatureMap,
    Orthogonalizer,
    OuterStep,
    RefreshPipeline,
    RefreshPolicy,
    ResidualSolver,
    Solver,
    WhiteningMap,
)
from kernos.core.lifecycle import Lifecycle
from kernos.core.plan import Buffer, Plan
from kernos.core.rng import Rng
from kernos.core.state import Bundle, Continuous, Discrete
from kernos.core.types import Array, asfloat64, check

__all__ = [
    "Array",
    "AnchorSampler",
    "AblationConfig",
    "AwarenessError",
    "Basis",
    "BudgetExceeded",
    "Buffer",
    "BudgetAccountant",
    "Bundle",
    "BasisConfig",
    "Continuous",
    "Discrete",
    "DriftMetric",
    "EmbeddingConfig",
    "TrainingDiagnostics",
    "FeatureMap",
    "FeatureFusion",
    "IllConditionedError",
    "MemoryLimitExceeded",
    "Lifecycle",
    "LocalFeatureMap",
    "Plan",
    "Orthogonalizer",
    "OuterStep",
    "RefreshPipeline",
    "RefreshPolicy",
    "ResidualSolver",
    "Rng",
    "RefreshConfig",
    "SolverConfig",
    "Solver",
    "ShapeError",
    "SolverEvent",
    "SolverStage",
    "RecoveryOutcome",
    "SolverRecovery",
    "TrainingCancelled",
    "TrainingTimeout",
    "TrainingConfig",
    "ValidationConfig",
    "WhiteningMap",
    "asfloat64",
    "check",
    "build_drift_metric",
    "build_embedder",
    "build_solver",
]
