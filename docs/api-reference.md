# Public API reference

This page is the compatibility boundary for users and extension authors. Names
listed here are supported public API; implementation modules and names beginning
with one underscore are not compatibility promises.

## Estimator

```python
from kernos import Kernos, SolverEvent
```

`Kernos` is a regressor with a scikit-learn-style interface. Its supported
operations are `fit`, `partial_fit`, `predict`, `score`, `get_params`, and
`set_params`. Training requires enough samples for the configured landmark and
validation capacities; configure `mbasis`, `abasis`, and `val_frac` together.

Fitted attributes use the scikit-learn trailing-underscore convention:
`bundle_`, `basis_`, `predictor_`, `diagnostics_`, `n_features_in_`, and
`lifecycle_`. Fitted state is published atomically; cancelled or timed-out
refits preserve the previous snapshot.

## Configuration and lifecycle

```python
from kernos import Buffer, Lifecycle
from kernos.core import (
    AblationConfig,
    BasisConfig,
    EmbeddingConfig,
    RefreshConfig,
    SolverConfig,
    TrainingConfig,
    ValidationConfig,
)
```

`Buffer.FULL`, `Buffer.STREAM`, and `Buffer.ADAPTIVE` select the supported
memory strategies. `Lifecycle` reports `UNFITTED`, `FITTING`, `FITTED`, or
`STREAMING`.

The grouped `*Config` classes are immutable views exposed from `Plan`; the
flat estimator constructor remains the compatibility layer for sklearn tools.
Core algorithm code uses descriptive `Plan` aliases such as
`representation_regularization`, `finite_difference_step`, `refresh_cooldown`,
`refresh_cost`, `anchor_mix`, `local_bandwidth`, and `local_neighbors`; the
short legacy fields remain accepted for compatibility.

## Diagnostics and errors

```python
from kernos import TrainingCancelled, TrainingTimeout
from kernos.core import (
    AwarenessError,
    BudgetExceeded,
    IllConditionedError,
    MemoryLimitExceeded,
    ShapeError,
    SolverEvent,
    SolverStage,
    RecoveryOutcome,
    SolverRecovery,
    TrainingDiagnostics,
)
```

`TrainingDiagnostics` is an immutable snapshot containing progress, refresh,
accepted/rejected sample counts, budget, memory, timing, conditioning and
solver-convergence information. Rejected streaming batches are observable but
never enter model state.
`SolverEvent` is a stable string-compatible enum for `cholesky_jitter`, `pcg_nonconverged`, and
`pcg_breakdown`. `SolverStage` identifies whether the most recent recorded
solve occurred during `initialize` or `refresh`. `RecoveryOutcome` and
`SolverRecovery` provide an immutable record of whether stabilization was not
needed, recovered, failed to converge, or failed with an exception. Domain failures inherit from
`AwarenessError`; `ShapeError` also inherits from `ValueError` for
scikit-learn-compatible input handling. `MemoryLimitExceeded` also inherits
from the built-in `MemoryError`, preserving conventional handling while
allowing callers to identify a Kernos preflight refusal.

## Extension protocols

Extension implementations should satisfy the smallest applicable protocol from
`kernos.core.interfaces`: `Basis`, `FeatureMap`, `LocalFeatureMap`, `Solver`,
`ResidualSolver`, `Orthogonalizer`, `FeatureFusion`, `AnchorSampler`,
`WhiteningMap`, `RefreshPipeline`, `OuterStep`, `BudgetAccountant`,
`RefreshPolicy`, or `DriftMetric`. The `LoopComponents` graph accepts these
contracts for dependency injection and fault isolation. Register strategy names
through the factory boundary in `kernos.core.factory`; algorithmic code should
not add new string conditionals.

## Visibility and compatibility

- No leading underscore: public and documented.
- One leading underscore: module-private implementation detail.
- Two leading underscores: reserved for name-mangling collision avoidance.
- Trailing underscore: fitted estimator attribute.
- Public API additions require tests, documentation and a compatibility note.
