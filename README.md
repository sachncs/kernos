<p align="center">
  <h1 align="center">Kernos</h1>
  <p align="center">Refresh-aware hybrid continuous-discrete low-rank kernel learning for scalable, adaptive kernel regression.</p>
  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
    <a href="https://github.com/sachncs/kernos/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/kernos/ci.yml?branch=master" alt="CI"></a>
    <a href="https://pypi.org/project/kernos/"><img src="https://img.shields.io/pypi/v/kernos" alt="PyPI"></a>
    <a href="https://github.com/sachncs/kernos/stargazers"><img src="https://img.shields.io/github/stars/sachncs/kernos" alt="Stars"></a>
  </p>
</p>

**Kernos** is a Python library that implements refresh-aware hybrid
continuous-discrete low-rank kernel regression. It separates model parameters
into continuous (updated every step) and discrete (refreshed adaptively)
groups, enabling efficient training with dynamic basis adaptation.

---

## Features

- **Explicit-feature kernel system** — PSD guarantee (`K = Phi Phi^T >= 0`) with rank bound `rank(K) <= r_g + m_l`
- **Global Nyström basis** — Landmark selection via k-means++ with soft-truncated spectral whitening
- **Local corrective features** — Residual-aware anchor sampling and k-NN sparse RBF features
- **Residual orthogonalization** — Ensures local features live in the global nullspace
- **Feature calibration and fusion** — Trace-based normalization and logistic gate balancing global/local contributions
- **Refresh controller** — Drift-aware trigger with cooldown, warmup, hysteresis, and amortized budget
- **Two memory modes** — Cached O(nm) and streamed O(m^2) for different scale regimes
- **Numerical stabilization** — Eigenvalue clipping, soft spectral truncation, and Cholesky with jitter fallback
- **Sklearn-compatible API** — `fit`, `predict`, `score` with `GridSearchCV` and `Pipeline` support

---

## Installation

### From PyPI

```bash
pip install kernos
```

### From source

```bash
git clone https://github.com/sachncs/kernos.git
cd kernos
pip install -e .
```

### With dev dependencies

```bash
pip install -e ".[dev]"
```

**Requirements**: Python >= 3.10, NumPy >= 1.24, SciPy >= 1.10, scikit-learn >= 1.3

---

## Quick Start

### Python API

```python
import numpy as np
from kernos import Kernos

# Generate synthetic data
rng = np.random.default_rng(42)
X_train = rng.standard_normal((200, 4))
y_train = X_train[:, 0] + 0.5 * X_train[:, 1] ** 2 + 0.1 * rng.standard_normal(200)

X_test = rng.standard_normal((50, 4))
y_test = X_test[:, 0] + 0.5 * X_test[:, 1] ** 2 + 0.1 * rng.standard_normal(50)

# Fit the model
from kernos.core.plan import Buffer

model = Kernos(
    dim=4,
    mbasis=32,
    abasis=8,
    ridge=1e-2,
    steps=50,
    seed=42,
)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"R^2 score: {model.score(X_test, y_test):.4f}")
```

### Memory Modes

```python
from kernos.core.plan import Buffer

# Cached (default) - O(nm) memory, simpler
model = Kernos(mode=Buffer.FULL)

# Streamed - O(m^2) memory, scales to larger datasets
model = Kernos(mode=Buffer.STREAM)
```

### Ablation Studies

```python
# Disable specific components for ablation
model = Kernos(
    noref=True,   # No discrete refreshes
    noorth=True,  # Skip orthogonalization
    nodiv=True,   # Remove diversity regularization
)
```

---

## Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dim` | 64 | Dimension of continuous embedding space |
| `mbasis` | 512 | Global basis rank budget (landmarks) |
| `abasis` | 128 | Local corrective rank budget (anchors) |
| `ridge` | 1e-3 | Ridge regularization parameter |
| `mode` | `Buffer.FULL` | `Buffer.FULL`, `Buffer.STREAM`, or `Buffer.ADAPTIVE` |
| `steps` | 1000 | Maximum training steps |
| `seed` | `None` | Random seed for reproducibility |

### Refresh Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `drift_hi` | 0.1 | Drift threshold to trigger refresh |
| `cool` | 50 | Minimum steps between refreshes |
| `warm` | 10 | Minimum step before first refresh |
| `gain` | 0.01 | Validation gain threshold scaled by refresh cost |

### Outer-Loop Optimizer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 1e-4 | Learning rate for gradient descent on R |
| `wr` | 0.0 | Frobenius regularizer weight |
| `worth` | 0.0 | Orthogonality penalty weight |
| `wdiv` | 0.0 | Diversity penalty weight |
| `fdeps` | 1e-5 | Finite-difference perturbation magnitude |

See [docs/getting-started.md](docs/getting-started.md) for detailed configuration options.

---

## API

| Symbol | Type | Description |
|--------|------|-------------|
| `Kernos` | class | Sklearn-compatible estimator |
| `Kernos.fit` | method | Fit the model on training data |
| `Kernos.predict` | method | Predict target values |
| `Kernos.score` | method | R² score on test data |
| `Kernos(mode=...)` | constructor arg | `Buffer.FULL`, `Buffer.STREAM`, or `Buffer.ADAPTIVE` |
| `Kernos(noref=...)` | constructor arg | Ablation toggle |

---

## Examples

### Basic regression with synthetic data

```python
import numpy as np
from kernos import Kernos

rng = np.random.default_rng(42)
X = rng.standard_normal((200, 4))
y = X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.1 * rng.standard_normal(200)

model = Kernos(seed=42).fit(X, y)
print(model.score(X, y))
```

### GridSearchCV integration

```python
from sklearn.model_selection import GridSearchCV
from kernos import Kernos

param_grid = {"mbasis": [16, 32, 64], "abasis": [4, 8, 16], "ridge": [1e-3, 1e-2]}
search = GridSearchCV(Kernos(seed=42), param_grid, cv=5)
search.fit(X_train, y_train)
print(search.best_params_)
```

See [examples/](examples/) for full evaluation scripts.

---

## Architecture

The method separates model parameters into two groups:

- **Continuous parameters** (`theta`, `R`): Updated every training step via gradient descent
- **Discrete parameters** (`Z`, `A`, `M_g`, `c_g`, `c_l`, `rho`): Refreshed only when drift exceeds a threshold

This hybrid approach makes the method efficient for streaming/large-batch settings, because the expensive discrete refresh is triggered adaptively rather than every step.

### Mathematical Guarantees

1. **PSD kernel**: `K = Phi Phi^T` is positive semidefinite
2. **Rank bound**: `rank(K) <= r_g + m_l`
3. **SPD normal equations**: `S = Phi^T Phi + lambda I` is symmetric positive definite
4. **Orthogonalization**: `Phi_g^T Phi_l_perp ~= 0` up to ridge regularization
5. **Calibration stability**: calibration scalars bounded away from zero

See [docs/architecture.md](docs/architecture.md) for full design rationale and extension points.

---

## Project Structure

```
kernos/
├── kernos/                    # SDK package
│   ├── __init__.py            # Public API exports (Kernos)
│   ├── _version.py            # PEP 440 version
│   ├── estimator.py           # Sklearn-compatible Kernos estimator
│   ├── numeric.py             # Eigenvalue clip, soft truncation, chol, pcg, drift
│   ├── linalg.py              # Linear-algebra helpers
│   ├── sample.py              # k-means++, farthest-point sampling
│   ├── core/                  # Types, errors, frozen Plan + state containers
│   │   ├── types.py           # Array, check
│   │   ├── error.py           # IllConditionedError
│   │   ├── plan.py            # Plan dataclass + Buffer enum
│   │   ├── state.py           # Continuous, Discrete, Bundle (frozen)
│   │   └── rng.py             # Rng helpers
│   ├── embed/                 # Dense embedders and projection matrix
│   │   ├── linear.py          # Linear
│   │   ├── identity.py        # Identity
│   │   ├── kernel.py          # Kernel (RBF)
│   │   └── projector.py       # Projector (normalize + R-projection)
│   ├── basis/                 # Nyström landmarks, whitening, random features
│   │   ├── nystrom.py         # Nystrom
│   │   ├── whitening.py       # Whitening (soft-truncated)
│   │   ├── random.py          # Random (RFF)
│   │   └── greedy.py          # Greedy landmark selector
│   ├── correct/               # Anchor sampling, RBF, orthogonalization
│   │   ├── sampler.py         # Residual-aware anchor sampler
│   │   ├── rbf.py             # k-NN sparse RBF features
│   │   └── orth/              # Ridge, Tikhonov projectors
│   ├── fuse/                  # Calibration, gate, fused-feature builder
│   │   ├── scaler.py          # Trace-based feature normalization
│   │   ├── gate.py            # Logistic sigmoid gate
│   │   └── fuse.py            # Fuse
│   ├── solver/                # Ridge regression (direct + iterative + Jacobi)
│   │   ├── direct.py          # Direct (Cholesky + jitter)
│   │   ├── iterative.py       # Iterative (PCG)
│   │   ├── woodbury.py        # Woodbury
│   │   ├── jacobi.py          # Jacobi preconditioner
│   │   └── equation.py        # Equation assembly
│   ├── cache/                 # Full / Stream / Adaptive accumulators
│   ├── policy/                # Drift, budget, refresh controller, policy
│   │   ├── budget.py          # Budget accountant
│   │   ├── policy.py          # Refresh trigger policy
│   │   ├── refresh.py         # Seven-step refresh pipeline
│   │   └── drift/             # Frobenius, Spectral drift metrics
│   ├── loop/                  # Training loop, callbacks, outer-step
│   │   ├── loop.py            # Loop
│   │   ├── callback.py        # Log, Snapshot, Profile
│   │   ├── outerstep.py       # Outerstep (SPSA)
│   │   └── loss.py            # Outer-loop loss terms
│   ├── predict/               # Mean prediction
│   └── bench/                 # Datasets, baselines, metrics, runner
│       ├── baseline.py        # Ridge, Nystrom, Random baselines
│       ├── dataset.py         # Synthetic regression generators
│       ├── metric.py          # rmse, mae, r2, maxerr, allmetrics
│       └── runner.py          # ExperimentRunner
├── tests/                     # Test suite (unit, numeric, integration)
├── examples/                  # Real-world evaluation examples
├── docs/                      # Documentation
└── pyproject.toml             # Build & tool config
```

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check kernos/ tests/
ruff format kernos/ tests/
mypy kernos/
```

### Code Style

- Line length: 100
- Quotes: double (`"`)
- Formatting: ruff (auto-format with `ruff format`)
- Type hints: required on all public signatures
- No semi-private naming (`_foo`) — all identifiers are public

### Commit Conventions

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add residual-aware anchor selection
fix: handle edge case in drift computation
docs: add comprehensive docstrings across all modules
refactor: convert semi-private attributes to public API
test: add parity tests for cached vs streamed memory
chore: update ruff config
```

---

## Testing

```bash
pytest tests/ -v                # Full suite
pytest tests/unit/ -v           # Unit tests only
pytest --cov=kernos            # With coverage
```

---

## Build

```bash
python -m build
```

---

## Release

1. Bump version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Commit with a `version:X.Y.Z` message
4. Tag and push — CI publishes to PyPI

---

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.10+ |
| Numerical | [NumPy](https://numpy.org/), [SciPy](https://scipy.org/) |
| Machine Learning | [scikit-learn](https://scikit-learn.org/) |
| Lint/Format | [ruff](https://docs.astral.sh/ruff/) |
| Type Check | [mypy](https://mypy-lang.org/) (strict) |
| Testing | [pytest](https://docs.pytest.org/) + pytest-cov |
| Build | [Hatchling](https://hatch.pypa.io/) |

---

## Roadmap

- **v0.0.2** — Current release: core implementation, sklearn API, test suite
- **v0.2.0** — GPU solver backend (CuPy), learned embedding functions
- **v1.0.0** — Stable API, PyPI release, streaming/incremental fit

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development setup
- Pull request process
- Coding standards
- Test expectations

## Code of Conduct

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md).
By participating you agree to abide by its terms.

## Security

Report vulnerabilities to **sachncs@gmail.com** — see [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE) © 2026 Sachin