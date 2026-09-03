<p align="center">
  <h1 align="center">Kernos</h1>
  <p align="center">Refresh-aware hybrid continuous-discrete low-rank kernel learning for scalable, adaptive kernel regression.</p>
  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
    <a href="https://github.com/sachncs/kernos/releases/latest"><img src="https://img.shields.io/github/v/release/sachncs/kernos" alt="Latest release"></a>
    <a href="https://github.com/sachncs/kernos/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/kernos/ci.yml?branch=master" alt="CI"></a>
    <a href="https://pypi.org/project/kernos/"><img src="https://img.shields.io/pypi/v/kernos" alt="PyPI"></a>
    <a href="https://github.com/sachncs/kernos/stargazers"><img src="https://img.shields.io/github/stars/sachncs/kernos" alt="Stars"></a>
    <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/code%20style-ruff-000000.svg" alt="Ruff"></a>
    <a href="https://mypy-lang.org/"><img src="https://img.shields.io/badge/type%20checked-mypy-blue.svg" alt="mypy"></a>
  </p>
</p>

---

## What is this?

**Kernos** is a Python library that implements refresh-aware hybrid
continuous-discrete low-rank kernel regression. It separates model
parameters into two groups:

- **Continuous parameters** — updated every training step via gradient descent.
- **Discrete parameters** — refreshed adaptively only when the basis drifts.

The result is efficient training with dynamic basis adaptation that
scales gracefully to large-batch and streaming settings.

---

## Who is this for?

You, if any of these sound like you:

- You're a researcher benchmarking **kernel methods** on tabular regression.
- You want an **adaptive basis** that refreshes landmarks only when it needs to, instead of every epoch.
- You use **scikit-learn** and want a drop-in estimator that supports `fit` / `predict` / `score` / `GridSearchCV`.
- You train on **large datasets** and need a memory mode that scales (cached `O(nm)` or streamed `O(m²)`).
- You're curious about **refresh-aware hybrid optimization** as a research direction.

---

## What can it do?

- **Explicit-feature kernel system** — PSD guarantee (`K = ΦΦᵀ ≥ 0`) with rank bound `rank(K) ≤ r_g + m_l`.
- **Global Nyström basis** — Landmark selection via k-means++ with soft-truncated spectral whitening.
- **Local corrective features** — Residual-aware anchor sampling and k-NN sparse RBF features.
- **Residual orthogonalization** — Local features live in the global nullspace.
- **Feature calibration and fusion** — Trace-based normalization and a logistic gate balancing global/local contributions.
- **Refresh controller** — Drift-aware trigger with cooldown, warmup, hysteresis, and amortized budget.
- **Two memory modes** — Cached `O(nm)` and streamed `O(m²)` for different scale regimes.
- **Numerical stabilization** — Eigenvalue clipping, soft spectral truncation, Cholesky with jitter fallback.
- **Sklearn-compatible API** — `fit`, `predict`, `score` with `GridSearchCV` and `Pipeline` support.

---

## Before you start

You'll need **Python 3.10 or newer**. Kernos depends on:

| Package | Minimum version | What it's for |
|---|---|---|
| NumPy | 1.24 | Array math |
| SciPy | 1.10 | Sparse / linalg helpers |
| scikit-learn | 1.3 | Estimator API surface |

The evaluation suite additionally depends on HuggingFace `datasets`
and `matplotlib` (install with the `examples` extra — see below).

If you're unsure which Python is on your machine, open a terminal and
type `python3 --version`. If the version starts with `3.10`, `3.11`,
or `3.12`, you're set.

---

## Installation

### Option 1 — From PyPI (fastest)

```bash
pip install kernos
```

### Option 2 — From source (recommended for development)

```bash
git clone https://github.com/sachncs/kernos.git
cd kernos
pip install -e .
```

### Option 3 — With dev + examples extras

```bash
pip install -e ".[dev,examples]"
```

> 💡 **The dot in `.[dev,examples]` is intentional.** It means
> "install this package and also the dev and examples extras." The
> square brackets are part of the command, not punctuation.

`dev` pulls in pytest, mypy, and ruff. `examples` pulls in HuggingFace
`datasets` and `matplotlib` for the evaluation suite and plots.

---

## Your first run — Python

Open a Python interpreter and try this:

```python
import numpy as np
from kernos import Kernos

# Synthetic regression: y depends on x0 and x1^2 with a little noise
rng = np.random.default_rng(42)
X_train = rng.standard_normal((200, 4))
y_train = X_train[:, 0] + 0.5 * X_train[:, 1] ** 2 + 0.1 * rng.standard_normal(200)
X_test = rng.standard_normal((50, 4))
y_test = X_test[:, 0] + 0.5 * X_test[:, 1] ** 2 + 0.1 * rng.standard_normal(50)

model = Kernos(
    dim=4,
    mbasis=24,
    abasis=4,
    lk=4,
    ridge=1e-2,
    steps=50,
    seed=42,
).fit(X_train, y_train)

print(f"R^2 score on held-out test set: {model.score(X_test, y_test):.4f}")
```

You'll see something like `R^2 score on held-out test set: 0.3…`. The
exact number depends on the seed; the model has fit successfully and
predicts better than the noise floor.

The full walk-through with explanations of every line lives in
[`docs/getting-started.md`](docs/getting-started.md).

---

## Your first run — CLI

After `pip install -e ".[dev,examples]"` you also get the
`kernos-eval` console script:

```bash
kernos-eval --datasets WineQuality --tiers Small --n_seeds 2 --output_dir results
kernos-analyze --results results/results.csv --output_dir results
```

This runs the full evaluation suite on a single HuggingFace dataset
at the `Small` budget tier, then produces:

- `results/results.md` — markdown tables
- `results/results.csv` — machine-readable metrics
- `results/pareto_WineQuality.png` — RMSE vs train time scatter
- `results/ablations_WineQuality_Small.png` — relative RMSE change per ablation
- `results/results_table.tex` — LaTeX table for papers

See [`examples/evaluation/README.md`](examples/evaluation/README.md)
for the full dataset / tier reference, and
[`docs/evaluation-protocol.md`](docs/evaluation-protocol.md) for the
experimental methodology.

---

## Configuration

Want to change something? Pass it as a constructor argument to
`Kernos(...)`. The most useful knobs:

| Parameter | Default | Plain English |
|---|---|---|
| `dim` | `64` | Dimension of the continuous embedding space. Bigger = more flexible but slower. |
| `mbasis` | `512` | Global basis rank (number of Nyström landmarks). |
| `abasis` | `128` | Local corrective rank (number of anchor points). |
| `ridge` | `1e-3` | Ridge regularization strength. Bigger = smoother fit. |
| `mode` | `Buffer.FULL` | Memory mode: `FULL` (cached) or `STREAM` (streamed, `O(m²)` memory). |
| `steps` | `1000` | Maximum number of training steps. |
| `seed` | `None` | Random seed for reproducibility. |

### Refresh controller

| Parameter | Default | Plain English |
|---|---|---|
| `drift_hi` | `0.1` | Drift threshold above which a refresh is considered. Smaller = refresh more often. |
| `cool` | `50` | Minimum number of steps between two refreshes. |
| `warm` | `10` | Minimum step before the *first* refresh is allowed. |
| `gain` | `0.01` | Validation gain threshold, scaled by per-refresh cost. |

### Outer-loop optimizer

| Parameter | Default | Plain English |
|---|---|---|
| `lr` | `1e-4` | Learning rate for gradient descent on `R`. |
| `wr` | `0.0` | Frobenius regularizer weight on `R`. |
| `worth` | `0.0` | Orthogonality penalty weight. |
| `wdiv` | `0.0` | Diversity penalty weight. |
| `fdeps` | `1e-5` | Finite-difference perturbation magnitude for SPSA. |

### Ablation flags

Each `Plan` boolean flag toggles one component on or off:

| Flag | Effect when `True` |
|---|---|
| `noref` | Skip all discrete refreshes (static basis after first init). |
| `nohyst` | Force hysteresis `b_t = 1` permanently. |
| `nocool` | Disable refresh cooldown. |
| `noresid` | Use coverage-only anchor sampling (no residual weighting). |
| `noorth` | Skip local orthogonalization. |
| `nodiv` | Set diversity penalty weight to zero. |
| `nofreeze` | Freeze calibration scalars after the first refresh. |

See [`docs/getting-started.md`](docs/getting-started.md) for the
complete reference.

---

## API

| Symbol | Type | Plain English |
|---|---|---|
| `Kernos` | class | The sklearn-compatible estimator. |
| `Kernos.fit(X, y)` | method | Fit on training data. |
| `Kernos.predict(X)` | method | Predict targets for new data. |
| `Kernos.score(X, y)` | method | R² score on test data. |
| `Kernos.partial_fit(X, y)` | method | Run a single continuous update (for streaming). |
| `Buffer` | enum | Memory mode: `FULL` (cached) or `STREAM` (streamed). |

---

## Examples

### Basic regression with synthetic data

```python
import numpy as np
from kernos import Kernos

rng = np.random.default_rng(42)
X = rng.standard_normal((200, 4))
y = X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.1 * rng.standard_normal(200)

model = Kernos(seed=42, mbasis=24, abasis=4, lk=4, steps=50).fit(X, y)
print(model.score(X, y))
```

### GridSearchCV integration

```python
from sklearn.model_selection import GridSearchCV
from kernos import Kernos

param_grid = {"mbasis": [16, 32, 64], "abasis": [4, 8, 16], "ridge": [1e-3, 1e-2]}
search = GridSearchCV(Kernos(seed=42, steps=50), param_grid, cv=5)
search.fit(X_train, y_train)
print(search.best_params_)
```

See [`examples/benchmarks/`](examples/benchmarks/) for small, runnable
demos and [`examples/evaluation/`](examples/evaluation/) for the full
real-world evaluation suite.

---

## Architecture

The method separates model parameters into two groups:

- **Continuous parameters** (`theta`, `R`): updated every training step via gradient descent.
- **Discrete parameters** (`Z`, `A`, `M_g`, `c_g`, `c_l`, `rho`): refreshed only when drift exceeds a threshold.

This hybrid approach makes the method efficient for
streaming/large-batch settings, because the expensive discrete
refresh is triggered adaptively rather than every step.

### Mathematical guarantees

1. **PSD kernel**: `K = ΦΦᵀ` is positive semidefinite.
2. **Rank bound**: `rank(K) ≤ r_g + m_l`.
3. **SPD normal equations**: `S = ΦᵀΦ + λI` is symmetric positive definite.
4. **Orthogonalization**: `Φ_gᵀ Φ_l_perp ≈ 0` up to ridge regularization.
5. **Calibration stability**: calibration scalars are bounded away from zero.

See [`docs/architecture.md`](docs/architecture.md) for full design
rationale and extension points, and
[`docs/design.md`](docs/design.md) for the mathematical foundations.

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
│   ├── embed/                 # Dense embedders and projection matrix
│   ├── basis/                 # Nyström landmarks, whitening, random features
│   ├── correct/               # Anchor sampling, RBF, orthogonalization
│   ├── fuse/                  # Calibration, gate, fused-feature builder
│   ├── solver/                # Ridge regression (direct + iterative + Jacobi)
│   ├── cache/                 # Full / Stream / Adaptive accumulators
│   ├── policy/                # Drift, budget, refresh controller
│   ├── loop/                  # Training loop, callbacks, outer-step
│   ├── predict/               # Mean prediction
│   └── bench/                 # Datasets, baselines, metrics, runner
├── examples/                  # Runnable code samples
│   ├── evaluation/            # Real-world evaluation suite (kernos-eval)
│   └── benchmarks/            # Small API demos
├── tests/                     # Test suite (unit, numeric, integration, examples)
├── docs/                      # Documentation
└── pyproject.toml             # Build & tool config
```

---

## Where to go next

**Users:**

- **[Getting Started](docs/getting-started.md)** — full installation and
  configuration walk-through.
- **[Examples](examples/README.md)** — index of `evaluation/` and `benchmarks/`.
- **[Benchmarks demos](examples/benchmarks/)** — small runnable scripts.
- **[Evaluation suite](examples/evaluation/)** — real-world benchmarks with `kernos-eval`.
- **[Evaluation Protocol](docs/evaluation-protocol.md)** — methodology.
- **[FAQ](docs/faq.md)** — common questions.

**Operators / Maintainers:**

- **[Architecture](docs/architecture.md)** — module-level design.
- **[Design](docs/design.md)** — mathematical foundations.
- **[Deployment](docs/deployment.md)** — publishing to PyPI.
- **[Contributing](CONTRIBUTING.md)** — pull request process.

---

## Development

```bash
pip install -e ".[dev,examples]"
pytest tests/ -v
ruff check kernos/ tests/ examples/
ruff format kernos/ tests/ examples/
mypy kernos/ examples/
```

### Code style

- Line length: 110 (`pyproject.toml`).
- Quotes: double (`"`).
- Formatting: `ruff format`.
- Type hints: required on all public signatures (mypy `--strict`).
- No semi-private naming (`_foo`) — all identifiers are public.

### Commit conventions

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add residual-aware anchor selection
fix: handle edge case in drift computation
docs: add comprehensive docstrings across all modules
refactor: convert semi-private attributes to public API
test: add parity tests for cached vs streamed memory
chore: update ruff config
```

### Testing

```bash
pytest tests/ -v                # Full suite
pytest tests/unit/ -v           # Unit tests only
pytest tests/examples/ -v       # Examples smoke tests
pytest --cov=kernos             # With coverage
```

### Build

```bash
python -m build
```

### Release

1. Bump version in `pyproject.toml`.
2. Update `CHANGELOG.md`.
3. Commit with a `version:X.Y.Z` message.
4. Tag and push — CI publishes to PyPI.

---

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.10+ |
| Numerical | [NumPy](https://numpy.org/), [SciPy](https://scipy.org/) |
| Machine Learning | [scikit-learn](https://scikit-learn.org/) |
| Evaluation datasets | [HuggingFace `datasets`](https://huggingface.co/docs/datasets) |
| Plots | [matplotlib](https://matplotlib.org/) |
| Lint / Format | [ruff](https://docs.astral.sh/ruff/) |
| Type Check | [mypy](https://mypy-lang.org/) (strict) |
| Testing | [pytest](https://docs.pytest.org/) + pytest-cov |
| Build | setuptools |

---

## Roadmap

- **v0.1.0** — Current release: core implementation, sklearn API, test suite.
- **v0.2.0** — GPU solver backend (CuPy), learned embedding functions.
- **v1.0.0** — Stable API, streaming/incremental `fit`.

---

## Contributing

Want to improve Kernos? See [CONTRIBUTING.md](CONTRIBUTING.md) for
how to set up a development environment and submit changes.

## Code of Conduct

We expect everyone to follow our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Security

Found a security issue? See [SECURITY.md](SECURITY.md) — please don't
open a public GitHub issue for security problems.

## License

[MIT](LICENSE) © 2026 Sachin.
