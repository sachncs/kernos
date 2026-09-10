# Architecture

This document describes the architecture of kernos, its module structure, and key design decisions.

## Overview

Kernos implements **refresh-aware hybrid continuous-discrete low-rank kernel learning**. The method separates model parameters into two groups:

- **Continuous parameters** (`theta`, `R`): Updated every training step via gradient descent.
- **Discrete parameters** (`Z`, `A`, `M_g`, `c_g`, `c_l`, `d`): Refreshed only when drift exceeds a threshold.

This hybrid approach makes the method efficient for streaming/large-batch settings, because the expensive discrete refresh is triggered adaptively rather than every step.

## Module Map

```
kernos/
├── core/               # Types, errors, frozen Plan + state containers
├── embed/              # Dense embedders and projection matrix
├── basis/              # Nyström landmarks, whitening, random features
├── correct/            # Anchor sampling, RBF, orthogonalization
├── fuse/               # Calibration, gate, fused-feature builder
├── solver/             # Ridge regression (direct + iterative + Jacobi + Woodbury)
├── cache/              # Full / Stream / Adaptive normal-equation accumulators
├── policy/             # Drift, budget, refresh controller, policy
├── loop/               # Training loop, callbacks, outer-step, loss
├── predict/            # Mean prediction
├── bench/              # Datasets, baselines, metrics, experiment runner
├── numeric.py          # Eigenvalue clip, soft truncation, chol, pcg, drift
├── linalg.py           # Linear-algebra helpers
├── sample.py           # k-means++, farthest-point sampling
└── estimator.py        # Sklearn-compatible Kernos estimator
```

## Pipeline

The data flow through the system follows these stages:

### 1. Embedding

Input `x` is mapped to a dense embedding via a linear layer:

```
e = x @ theta + bias
```

### 2. Projection

Embeddings are L2-normalized and projected through the learned matrix `R`:

```
u = R * normalize(e)
```

### 3. Global Basis (Nyström)

Landmarks `Z` are selected via k-means++. A whitened RBF kernel feature map is constructed:

```
phi_g(u) = k(u, Z) @ M_g
```

where `M_g` is the soft-truncated spectral whitening map.

### 4. Local Corrective

Residuals from a global-only ridge are computed. Anchors `A` are selected via residual-aware sampling. k-NN sparse RBF features are built, then orthogonalized against the global subspace:

```
phi_l_perp = (I - P_g) @ phi_l
```

### 5. Calibration and Fusion

Global and local features are scaled by trace-based calibration constants, then fused via a logistic gate:

```
rho = sigma(a)
phi = [sqrt(rho) * c_g * phi_g, sqrt(1-rho) * c_l * phi_l_perp]
```

### 6. Ridge Regression

Normal equations are solved:

```
(Phi^T @ Phi + lambda * I) w = Phi^T @ y
```

via Cholesky (direct) or PCG (iterative).

### 7. Refresh Controller

Drift is computed as:

```
drift = ||R_t - R_{t_r}||_F / ||R_{t_r}||_F
```

A refresh triggers only when: drift > threshold AND cooldown elapsed AND warmup passed AND hysteresis active AND validation gain exceeds budget-scaled cost.

## Key Design Decisions

### Immutability of State

`Continuous`, `Discrete`, and `Bundle` are frozen dataclasses with `replace` methods (via `dataclasses.replace`). This makes refresh boundaries explicit and avoids surprising mutations during the discrete pipeline.

### Duck-Typed Interfaces

Core abstractions (`Embed`, `Basis`, `Refresh`, `Solver`, `Drift`, `Cache`) are consumed via the same `forward` / `run` / `solve` method names rather than `typing.Protocol`. This keeps modules decoupled without the ceremony of explicit protocol declarations.

### Numerical Hardening

All numerical thresholds are centralized as `stab_*` fields on `Plan`:

| Parameter | Purpose |
|-----------|---------|
| `stab_tau` | Eigenvalue floor for soft truncation |
| `stab_alpha` | Dataset-scale epsilon for whitening stability |
| `stab_jitter` | Initial jitter for Cholesky-with-fallback |
| `stab_jitter_retry` | Maximum Cholesky fallback attempts |
| `stab_eta` | Ridge regularizer for orthogonalization matrix invertibility |
| `stab_kappa` | Maximum acceptable condition number |

### Cached vs. Streamed Memory

- **Cached**: Stores the full `Phi` matrix. Simpler, enables direct normal-equation construction. O(nm) memory.
- **Streamed**: Accumulates `S` and `b` online. Reduces memory to O(m^2). Parity tests confirm both modes produce identical coefficients (see `tests/unit/test_cache.py::TestAdaptive::test_parity_full`).

## Extension Points

### Custom Embedder

Subclass or compose any object that exposes `forward(X) -> ndarray` and a `params()` pair; pass it into `Loop.initialize` or override `Kernos.build_plan` to swap the `Linear` embedder.

### Alternative Refresh Policy

Replace `Policy.decide` with a custom decision rule and pass it into `Loop.__init__` in place of the default `Policy(plan, drift_value)`.

### Alternative Solver

`kernos.solver` ships `Direct` (Cholesky), `Iterative` (PCG), `Woodbury`, and `Jacobi`. All expose the same `solve(phi, y)` signature so the `Loop` can be parameterised.

## Testing Strategy

The test suite is organized in three tiers:

| Tier | Directory | Purpose |
|------|-----------|---------|
| Unit | `tests/unit/` | Shapes, API contracts, error paths, basic correctness |
| Numerical | `tests/numeric/` | Paper invariants (PSD, SPD, rank bounds, orthogonality) |
| Integration | `tests/integration/` | End-to-end parity, refresh behavior, convergence |

Coverage is enforced at 80% (currently ~91% on full runs).
