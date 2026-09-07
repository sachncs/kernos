# Examples

Runnable code that uses Kernos on real and synthetic data.

The examples tree is organised into two sub-packages:

- **`examples/evaluation/`** — full real-world evaluation suite
  (multiple datasets, budget tiers, baselines, ablations, plots,
  LaTeX export).
- **`examples/benchmarks/`** — small, runnable demos showcasing the
  public API.

---

## Before you start

The evaluation suite depends on HuggingFace `datasets` and
`matplotlib`; install both with the `examples` extra:

```bash
pip install -e ".[examples]"
```

Or for development:

```bash
pip install -e ".[dev,examples]"
```

The benchmark demos only need the core `kernos` package.

---

## Quick look — benchmarks

Each demo is a single self-contained script. Run any of them with
`python -m examples.benchmarks.<name>`:

| Demo | What it shows |
|---|---|
| `quickstart` | Minimal fit → predict on synthetic data. |
| `compare_baselines` | Kernos vs Ridge vs Nystrom vs RFF. |
| `memory_modes` | `Buffer.FULL` (cached) vs `Buffer.STREAM`. |
| `ablation_walk` | One ablation flag at a time. |
| `grid_search_demo` | `GridSearchCV` integration. |

```bash
python -m examples.benchmarks.quickstart
python -m examples.benchmarks.compare_baselines
python -m examples.benchmarks.memory_modes
python -m examples.benchmarks.ablation_walk
python -m examples.benchmarks.grid_search_demo
```

Outputs land in `examples/benchmarks/results/`.

---

## Quick look — evaluation suite

Two console scripts are exposed after install:

```bash
# Run the suite
kernos-eval --datasets WineQuality --tiers Small --n_seeds 2 --output_dir results

# Then post-process the CSV into plots and a LaTeX table
kernos-analyze --results results/results.csv --output_dir results
```

Or run them as modules:

```bash
python -m examples.evaluation.cli --datasets all --tiers Small --n_seeds 2
python -m examples.evaluation.analyze_cli --results results/results.csv
```

See [`examples/evaluation/README.md`](evaluation/README.md) for the
full dataset and tier reference, and
[`docs/evaluation-protocol.md`](../docs/evaluation-protocol.md) for
the experimental methodology.

---

## Where to go next

- [`examples/benchmarks/README.md`](benchmarks/README.md) — quick API demos.
- [`examples/evaluation/README.md`](evaluation/README.md) — full eval suite reference.
- [`docs/evaluation-protocol.md`](../docs/evaluation-protocol.md) — experimental methodology.
- [`docs/getting-started.md`](../docs/getting-started.md) — installation and configuration walk-through.
- [`docs/architecture.md`](../docs/architecture.md) — module-level design.
