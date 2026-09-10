# Design Document

This document describes the architecture, invariants, and extension points of
the kernos implementation.

## Overview

Kernos is a hybrid continuous-discrete learner for large-scale kernel
regression. The key idea is to separate representation parameters (continuous:
`theta`, `R`) from basis parameters (discrete: `Z`, `A`, `M_g`, `c_g`, `c_l`,
`d`). Continuous parameters are updated every step; discrete parameters are
refreshed only when drift exceeds a threshold, making the method efficient
for streaming or large-batch settings.

## Module Map

### `kernos.core`

Frozen configuration (`Plan`), state containers (`Continuous`, `Discrete`,
`Bundle`), shared types and protocols, and domain exceptions.

### `kernos.embed`

- `Linear`, `Identity`, `Kernel`: Maps input `x` to a dense continuous embedding.
- `Projector`: Applies the learned projection matrix `R` to embeddings.

### `kernos.basis`

- `Nystrom`: Selects landmarks `Z` and builds a whitened global feature map
  `phi_g(u) = k(u, Z) M_g`.
- `Whitening`: Implements soft-truncated spectral whitening with
  eigenvalue clipping and epsilon scaling.
- `Random`: Random Fourier feature map.
- `Greedy`: Greedy landmark selector.

### `kernos.correct`

- `Sampler`: Chooses anchors `A` by blending coverage and residual weights.
- `Rbf`: Builds k-NN sparse radial features.
- `Orth` (`Ridge`, `Tikhonov`): Projects local features into the global
  nullspace: `Phi_l_perp = (I - P_g) Phi_l`.

### `kernos.fuse`

- `Fuse`: Calibrates and gates global and local features:
  `phi = [sqrt(rho) * c_g * phi_g, sqrt(1-rho) * c_l * phi_l_perp]`.
- `Gate`: Logistic sigmoid gate `rho = sigma(a)`.
- `Scaler`: Trace-based feature normalization.

### `kernos.solver`

- `Direct`: Solves normal equations via Cholesky with conditioning
  checks and jitter fallback.
- `Iterative`: PCG with diagonal preconditioner for large `m`.
- `Woodbury`: Low-rank Woodbury solve.
- `Jacobi`: Diagonal Jacobi preconditioner.

### `kernos.cache`

- `Full`: Stores `Phi` explicitly; normal equations via
  matrix multiplication (`O(nm)`).
- `Stream`: Accumulates `S = Phi^T Phi` and `b = Phi^T y`
  directly (`O(m^2)`).
- `Adaptive`: Switches between `Full` and `Stream` based on a sample count.

### `kernos.policy`

- `Budget`: Budget accountant.
- `Policy`: Refresh trigger policy combining drift, cooldown, warmup,
  hysteresis, and budget.
- `Refresh`: Full discrete refresh pipeline from projected embeddings.
- `drift` (`Frobenius`, `Spectral`): Drift metrics.

### `kernos.loop`

- `Loop`: Main loop integrating initialization, continuous updates,
  refresh decisions, and evaluation.
- `Outerstep`: Outer-loop bilevel step including ridge loss,
  orthogonality penalty, and diversity penalty.
- `Callback` (`Log`, `Snapshot`, `Profile`): Logging and checkpoint hooks.
- `Loss`: Outer-loop loss terms.

### `kernos.predict`

- `Predict`: Mean prediction from fused features.

### `kernos.bench`

- `dataset.py`: Synthetic benchmarks (linear, polynomial, high-dimensional,
  heteroscedastic).
- `baseline.py`: Ridge, Nyström ridge, and random Fourier feature baselines.
- `metric.py`: RMSE, MAE, R^2, and max absolute error.
- `runner.py`: Reproducible experiment runner with timing.

### `kernos.estimator`

- `Kernos`: Sklearn-compatible public API wrapping `Loop`.

## Key Design Decisions

### Immutability of State

`Continuous`, `Discrete`, and `Bundle` are frozen dataclasses with `replace`
methods (delegating to `dataclasses.replace`). This makes refresh boundaries
explicit and avoids surprising mutations during the discrete pipeline.

### Duck-Typed Interfaces

Core abstractions (`Embed`, `Basis`, `Refresh`, `Solver`, `Drift`, `Cache`)
are consumed via the same `forward` / `run` / `solve` method names rather
than `typing.Protocol`. This keeps modules decoupled without the ceremony of
explicit protocol declarations.

### Numerical Hardening

All numerical thresholds are centralised as `stab_*` fields on `Plan`:

- `stab_tau`: Eigenvalue floor for soft truncation.
- `stab_alpha`: Dataset-scale epsilon for whitening stability.
- `stab_eps`: Calibration-scale floor.
- `stab_lmin`: Floor on ridge regularization for SPD guarantees.
- `stab_eta`: Ridge regularizer for orthogonalization matrix invertibility.
- `stab_kappa`: Maximum acceptable condition number.

These defaults were chosen to balance accuracy and stability on synthetic
benchmarks.

### Cached vs. Streamed Memory

Cached mode stores the full `Phi` matrix, which is simpler and enables direct
normal-equation construction. Streamed mode accumulates `S` and `b` online,
reducing memory from `O(nm)` to `O(m^2)`. Parity tests confirm both modes
produce identical coefficients when given the same data and seed
(`tests/unit/test_cache.py::TestAdaptive::test_parity_full`).

### Refresh Pipeline

The refresh pipeline is intentionally modular:

1. Landmark selection (k-means++).
2. Whitening map construction (eigenvalue clip + soft truncation).
3. Residual computation (current predictor minus targets).
4. Anchor sampling (coverage/residual blend).
5. Local sparse features (k-NN RBF).
6. Orthogonalization (residual projection).
7. Calibration (scalar normalization).
8. Fusion gate estimation.

Each step can be tested in isolation, and the pipeline can be extended with
alternative sampling or whitening strategies.

## Extension Points

### Custom Embedder

Any object that exposes `forward(X) -> ndarray` and a `params()` pair can be
swapped into `Loop.initialize` (or wrapped by overriding `Kernos.build_plan`).
The reference implementations are `Linear`, `Identity`, and `Kernel` under
`kernos.embed`.

### Alternative Refresh Policy

Replace `Policy.decide` with a custom decision rule and pass it into
`Loop.__init__` in place of the default `Policy(plan, drift_value)`.

### Alternative Solver

`kernos.solver` ships `Direct` (Cholesky), `Iterative` (PCG), `Woodbury`, and
`Jacobi`. All expose the same `solve(phi, y)` signature so the `Loop` can be
parameterised.

## Testing Strategy

The test suite is organized in three tiers:

1. **Unit tests** (`tests/unit/`): Verify shapes, API contracts, error paths,
   and basic correctness for every public function.
2. **Numerical invariant tests** (`tests/numeric/`): Verify paper guarantees
   (PSD, SPD, rank bounds, orthogonality, calibration stability) with
   randomized inputs and tight tolerances.
3. **Integration tests** (`tests/integration/`): End-to-end parity,
   refresh trigger behavior, and training loop convergence on synthetic data.

Coverage is enforced at 80% (currently ~91% on full runs), with omissions for
the evaluation module since it is benchmark scaffolding rather than core
library code.
