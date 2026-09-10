# Frequently Asked Questions

## General

### What is Kernos?

Kernos is a research implementation of **refresh-aware hybrid continuous-discrete low-rank kernel learning** for scalable, adaptive kernel regression. It combines a continuously updated embedding projection with a discrete refresh pipeline for the kernel basis.

### When should I use Kernos?

Kernos is designed for:

- Large-scale kernel regression where standard Nyström methods are too slow
- Streaming or online learning settings where data arrives sequentially
- Problems requiring adaptive kernel basis updates without full recomputation
- Research into hybrid continuous-discrete learning methods

### How does Kernos compare to standard Nyström ridge regression?

Standard Nyström methods rebuild the kernel basis from scratch each time. Kernos separates continuous parameters (updated every step) from discrete parameters (refreshed only when drift exceeds a threshold). This makes it much more efficient for streaming settings.

## Installation

### What Python versions are supported?

Python 3.10, 3.11, and 3.12 are officially supported and tested in CI.

### What are the dependencies?

- NumPy >= 1.24
- SciPy >= 1.10
- scikit-learn >= 1.3

### How do I install for development?

```bash
git clone https://github.com/sachncs/kernos.git
cd kernos
pip install -e ".[dev]"
```

## Usage

### How do I choose between memory modes?

- **Cached** (`mode=Buffer.FULL`): Stores the full normal-equation inputs. Use when memory is not a constraint and you want simplicity.
- **Streamed** (`mode=Buffer.STREAM`): Accumulates `(S, b)` directly. Use when `n` is very large and you cannot afford O(nm) memory.
- **Adaptive** (`mode=Buffer.ADAPTIVE`): Switches from cached to streamed once a sample-count threshold is crossed.

### What do the ablation flags do?

Ablation flags disable specific components for ablation studies:

| Flag | Effect |
|------|--------|
| `noref` | Never refresh the discrete basis |
| `nohyst` | Always consider refresh (ignore hysteresis) |
| `nocool` | Set effective cooldown to zero |
| `noresid` | Use coverage-only anchor sampling |
| `noorth` | Skip local feature orthogonalization |
| `nodiv` | Remove diversity regularization |
| `nofreeze` | Keep calibration updating after first refresh |

### How do I set a random seed for reproducibility?

Pass `seed=<integer>` to the constructor:

```python
model = Kernos(seed=42)
```

### Can I use custom embedding functions?

Yes. Any object that exposes a `forward(X) -> ndarray` method with a finite-difference steppable outer objective can be dropped in as the embedder. See `kernos/embed/` for the existing `Linear`, `Kernel`, and `Identity` reference implementations.

## Performance

### How many landmarks (`mbasis`) should I use?

General guidelines:

| Dataset size | Recommended `mbasis` |
|-------------|-----------------------|
| < 1K samples | 32 - 64 |
| 1K - 10K samples | 128 - 512 |
| 10K - 100K samples | 512 - 2048 |
| > 100K samples | 2048+ |

Start small and increase if needed. More landmarks = better approximation but higher cost.

### How do I reduce training time?

- Reduce `steps`
- Reduce `mbasis` (global landmarks) and `abasis` (local anchors)
- Use `noref=True` to skip discrete refresh entirely
- Increase `cool` to refresh less often

## Troubleshooting

### I get a conditioning warning. What does it mean?

The solver detected that the normal-equation matrix is ill-conditioned (condition number exceeds `stab_kappa`). This can happen when:

- `ridge` is too small
- Features are nearly collinear
- The dataset has very different scales

Try increasing `ridge` or preprocessing your data (e.g., standard scaling).

### The model seems to not converge. What should I try?

- Increase `steps`
- Decrease `ridge` (but watch for conditioning issues)
- Adjust `drift_hi` (lower = more frequent refreshes)
- Check that your data is properly scaled
- Try different `dim` and `mbasis` values

### How do I report bugs?

Please open an issue on GitHub using the bug report template. Include:

- A minimal reproducible example
- Full error traceback
- Python and package versions
- OS information

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development setup, coding standards, and pull request guidelines.
