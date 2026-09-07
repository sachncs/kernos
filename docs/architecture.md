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

`ContinuousState`, `DiscreteState`, and `FullState` are frozen dataclasses with `copy_with` methods. This makes refresh boundaries explicit and avoids surprising mutations during the discrete pipeline.

### Protocols over Inheritance

Core interfaces (`Embedder`, `RidgeSolver`, `MemoryAccumulator`, `RefreshPolicy`) are defined as `typing.Protocol`. This keeps modules decoupled and makes swapping implementations straightforward.

### Numerical Hardening

All numerical thresholds are centralized in `NumericsConfig`:

| Parameter | Purpose |
|-----------|---------|
| `tau_eig` | Eigenvalue floor for soft truncation |
| `alpha_epsilon` | Dataset-scale epsilon for whitening stability |
| `epsilon_c` | Minimum calibration scale to prevent feature collapse |
| `lambda_min` | Floor on ridge regularization for SPD guarantees |
| `eta_o` | Ridge regularizer for orthogonalization matrix invertibility |
| `kappa_threshold` | Maximum acceptable condition number |

### Cached vs. Streamed Memory

- **Cached**: Stores the full `Phi` matrix. Simpler, enables direct normal-equation construction. O(nm) memory.
- **Streamed**: Accumulates `S` and `b` online. Reduces memory to O(m^2). Parity tests confirm both modes produce identical coefficients.

## Extension Points

### Custom Embedder

Implement the `Embedder` protocol and pass it into a custom `TrainingLoop` initializer:

```python
class MyEmbedder:
    def embed(self, X: np.ndarray) -> np.ndarray:
        # Your embedding logic
        ...
```

### Alternative Refresh Policy

Implement the `RefreshPolicy` protocol and replace the default `should_refresh` logic in `TrainingLoop.maybe_refresh`.

### GPU Solver

Replace `DirectRidgeSolver` with a CuPy-backed solver that implements the `RidgeSolver` protocol. The `Loop` and `Kernos` will work unchanged.

## Testing Strategy

The test suite is organized in three tiers:

| Tier | Directory | Purpose |
|------|-----------|---------|
| Unit | `tests/unit/` | Shapes, API contracts, error paths, basic correctness |
| Numerical | `tests/numeric/` | Paper invariants (PSD, SPD, rank bounds, orthogonality) |
| Integration | `tests/integration/` | End-to-end parity, refresh behavior, convergence |

Coverage is enforced at 80% (currently ~91% on full runs).
