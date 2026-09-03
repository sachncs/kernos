# Evaluation Protocol

This document specifies how Kernos is benchmarked in
`examples/evaluation/`. It defines the datasets, the budget tiers,
the metrics, the ablations, and the reproducibility defaults.

The suite is driven by two console scripts:

```bash
kernos-eval     # run experiments
kernos-analyze  # post-process results into plots and LaTeX
```

---

## Datasets

We cover three dataset families:

| Family | Loader location | Examples |
|---|---|---|
| scikit-learn built-ins | `datasets/sources/sklearn_sources.py` | `CaliforniaHousing`, `Diabetes` |
| OpenML / UCI tabular | `datasets/sources/openml_sources.py` | `Protein`, `Kin8nm`, `YearPredictionMSD`, `NYCTaxi` |
| HuggingFace tabular | `datasets/sources/huggingface_sources.py` | `WineQuality`, `Abalone`, `Superconduct`, `HouseSales`, `Elevators`, `CPUActivity`, `Diamonds` |
| Synthetic | `datasets/sources/synthetic_sources.py` | `SyntheticSpatial` (2D smooth global + local high-freq field) |

All loaders return `(X, y)` as `np.ndarray[float64]` and drop rows
with non-finite entries.

---

## Splits and Preprocessing

- Train / val / test fractions: **70 / 15 / 15** (`runner/splits.py`).
- Standard scaling on both `X` and `y`, fit on train only.
- Random permutation seeded per run; the seed is recorded in the
  per-seed CSV row.

---

## Budget Tiers

Three preset tiers control the rank budget and training horizon:

| Tier | `mbasis` | `abasis` | `steps` | `dim` | `total_refresh_budget` |
|------|----------|----------|---------|-------|------------------------|
| Small | 128 | 32 | 200 | 16 | 5.0 |
| Medium | 512 | 128 | 1,000 | 64 | 20.0 |
| Large | 2,048 | 512 | 2,000 | 128 | 50.0 |

`refresh_cost` is fixed at `1.0` per refresh. `Buffer.FULL` (cached
`O(nm)`) is the default memory mode.

---

## Models

### Main method

**Kernos** — refresh-aware hybrid kernel with continuous `R`
optimization, local corrective features in the global nullspace,
drift-aware refresh controller, and amortized refresh budget.

### Baselines

| Baseline | Class | Description |
|---|---|---|
| `Ridge` | `kernos.bench.baseline.Ridge` | Dense exact ridge in the original feature space. |
| `Nystrom` | `kernos.bench.baseline.Nystrom` | Static Nyström feature ridge (no refresh, no local corrective). |
| `RFF` | `kernos.bench.baseline.Random` | Random Fourier Features ridge. |

### Ablations

Each ablation is implemented as a `Plan` boolean flag (see
`examples/evaluation/models/ablations.py`):

| Display name | Flag | Disabled component |
|---|---|---|
| `K-NoRefresh` | `noref=True` | Discrete refreshes |
| `K-NoHysteresis` | `nohyst=True` | Hysteresis (b_t = 1 permanently) |
| `K-NoCooldown` | `nocool=True` | Refresh cooldown |
| `K-NoResidAnchors` | `noresid=True` | Residual-aware anchor sampling |
| `K-NoOrthog` | `noorth=True` | Local orthogonalization |
| `K-NoDivPenalty` | `nodiv=True` | Diversity penalty |
| `K-NoFreeze` | `nofreeze=True` | Calibration freezing after first refresh |

---

## Hyperparameter Tuning

For each (dataset, model, seed) triple we tune the ridge `λ` over a
log-spaced grid via :func:`tune_lambda_reg`:

| Model | λ grid |
|---|---|
| Kernos + ablations | `[1e-4, 1e-3, 1e-2, 1e-1]` |
| Ridge / Nystrom / RFF | `[1e-4, 1e-3, 1e-2, 1e-1, 1.0]` |

Selection criterion: lowest validation RMSE.

---

## Metrics

Computed per single run (`reporting/result_record.py`):

| Metric | Symbol | Description |
|---|---|---|
| RMSE | `rmse` | Root mean squared error. |
| MAE | `mae` | Mean absolute error. |
| R² | `r2` | Coefficient of determination. |
| Train time | `train_time_sec` | Wall-clock fit time. |
| Predict time | `predict_time_sec` | Wall-clock predict time on test set. |
| Peak memory | `peak_mem_mb` | Max resident memory via `tracemalloc`. |
| Refresh count | `refresh_count` | Number of discrete refreshes triggered. |
| Condition proxy | `condition_proxy` | `κ(ΦᵀΦ + λI)` eigenvalue ratio. |

### Stability-over-Refits

For each (dataset, model, tier) cell we also report the across-seed
variance of RMSE, train time, and condition proxy.

---

## Reproducibility

- Default seeds: `[42, 43, 44, 45, 46]` (controlled by `--n_seeds`).
- CPU only; no GPU dependency.
- Software versions are recorded in the environment used to run.
- Stopping criterion: `max_steps`; eval every `max_steps // 20`.

---

## Outputs

The runner writes:

- `results.md` — markdown tables (mean ± std metrics; stability).
- `results.csv` — machine-readable, one row per (dataset, model, tier).
- `pareto_{dataset}.png` — RMSE vs train time scatter.
- `ablations_{dataset}_{tier}.png` — relative RMSE change per ablation.
- `results_table.tex` — LaTeX table of Kernos + baselines.
