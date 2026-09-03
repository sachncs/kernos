# Getting Started

This guide walks you through installing and using Kernos for the first time.

## Prerequisites

- Python >= 3.10
- pip or conda package manager

## Installation

### From source (recommended for development)

```bash
git clone https://github.com/sachncs/kernos.git
cd kernos
pip install -e ".[dev]"
```

### Verify installation

```python
import kernos
print(kernos.__version__)
```

## Quick Start

### Basic regression

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

### Using different memory modes

```python
from kernos.core.plan import Buffer

# Cached mode (default) - stores full feature matrix, O(nm) memory
model_cached = Kernos(mode=Buffer.FULL)

# Streamed mode - accumulates normal equations directly, O(m^2) memory
model_streamed = Kernos(mode=Buffer.STREAM)
```

### Using callbacks for monitoring

```python
from kernos.loop.callback import Log

# Log every 10 steps
model = Kernos(log_every=10, steps=100)
model.fit(X_train, y_train)
```

## Configuration

### Key hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dim` | 64 | Dimension of continuous embedding space |
| `mbasis` | 512 | Global basis rank (landmarks) |
| `abasis` | 128 | Local corrective rank (anchors) |
| `ridge` | 1e-3 | Ridge regularization parameter |
| `steps` | 1000 | Maximum training steps |
| `seed` | None | Random seed for reproducibility |

### Refresh parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `drift_hi` | 0.1 | Drift threshold to trigger refresh |
| `cool` | 50 | Minimum steps between refreshes |
| `warm` | 10 | Minimum step before first refresh |
| `gain` | 0.01 | Validation gain threshold |
| `budget` | inf | Total amortized refresh budget |

### Ablation flags

| Parameter | Default | Description |
|-----------|---------|-------------|
| `noref` | False | Skip all discrete refreshes |
| `nohyst` | False | Force hysteresis = 1 permanently |
| `nocool` | False | Set effective cooldown to 0 |
| `noresid` | False | Use coverage-only sampling |
| `noorth` | False | Skip local orthogonalization |
| `nodiv` | False | Set diversity penalty to 0 |
| `nofreeze` | False | Freeze calibration after first refresh |

## Next steps

- Read the [Architecture Guide](architecture.md) for module details
- Check the [Design Document](design.md) for mathematical foundations
- Review `examples/evaluation/` for the real-world evaluation suite
  (see [Evaluation Protocol](evaluation-protocol.md) for the methodology)
- Browse `examples/benchmarks/` for small, runnable API demos
- See the [FAQ](faq.md) for common questions
