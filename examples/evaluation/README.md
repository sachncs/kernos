# Real-World Evaluation Suite for Kernos

This sub-package implements the full evaluation protocol (see
[`docs/evaluation-protocol.md`](../../docs/evaluation-protocol.md))
on a diverse collection of real-world, HuggingFace, and synthetic
datasets.

---

## Structure

```
evaluation/
├── README.md                 # This file
├── __init__.py               # Package marker; re-exports DATASET_REGISTRY
├── cli.py                    # kernos-eval entry point
├── analyze_cli.py            # kernos-analyze entry point
├── datasets/
│   ├── registry.py           # DATASET_REGISTRY (name → loader)
│   └── sources/
│       ├── sklearn_sources.py      # CaliforniaHousing, Diabetes
│       ├── openml_sources.py       # Protein, Kin8nm, YearPredictionMSD, NYCTaxi
│       ├── huggingface_sources.py  # WineQuality, Abalone, ...
│       └── synthetic_sources.py    # SyntheticSpatial
├── models/
│   ├── tiers.py              # BudgetTier + SMALL/MEDIUM/LARGE
│   ├── factories.py          # make_kernos, make_nystrom, make_rff, make_ridge
│   └── ablations.py          # ABLATION_PRESETS
├── runner/
│   ├── splits.py             # preprocess_and_split
│   ├── single.py             # run_single_kernos, run_single_baseline
│   ├── diagnostics.py        # compute_condition_proxy, tune_lambda_reg
│   └── suite.py              # run_experiment_suite
├── reporting/
│   ├── result_record.py      # SingleRunResult, ExperimentResult
│   ├── markdown.py           # results_to_markdown, stability_to_markdown
│   ├── csv_writer.py         # write_results_csv
│   ├── plots.py              # plot_pareto_fronts, plot_ablation_bars
│   └── latex.py              # export_latex_table
└── results/                  # Output directory (created at runtime)
    ├── results.md
    ├── results.csv
    ├── pareto_{dataset}.png
    ├── ablations_{dataset}_{tier}.png
    └── results_table.tex
```

---

## Datasets

### Sklearn / OpenML (UCI tabular)

| Dataset | Source | n | d | Type |
|---------|--------|---|---|------|
| `CaliforniaHousing` | `sklearn.datasets` | ~20,640 | 8 | Real-world regression |
| `Diabetes` | `sklearn.datasets` | 442 | 10 | Medical regression |
| `Protein` | OpenML 195 | 159 | 15 | UCI tabular |
| `Kin8nm` | OpenML 189 | 8,192 | 8 | UCI tabular |
| `YearPredictionMSD` | OpenML 227 | ~8,192 | 12 | UCI tabular |
| `NYCTaxi` | OpenML 42729 | ~581,835 | 18 | Large-scale benchmark |

### HuggingFace (`inria-soda/tabular-benchmark`)

| Dataset | Config | n | d | Notes |
|---------|--------|---|---|-------|
| `WineQuality` | `reg_num_wine_quality` | ~6,497 | 11 | Tabular regression |
| `Abalone` | `reg_num_abalone` | ~4,177 | 7 | Tabular regression |
| `Superconduct` | `reg_num_superconduct` | ~21,263 | 80 | High-dimensional regression |
| `HouseSales` | `reg_num_house_sales` | ~21,613 | 15 | Tabular regression |
| `Elevators` | `reg_num_elevators` | ~16,599 | 18 | Tabular regression |
| `CPUActivity` | `reg_num_cpu_act` | ~8,192 | 21 | Tabular regression |
| `Diamonds` | `reg_cat_diamonds` | ~53,940 | 9 | Categorical features encoded |

### Synthetic

| Dataset | Generator | n | d | Purpose |
|---------|-----------|---|---|---------|
| `SyntheticSpatial` | `load_synthetic_spatial` | configurable | 2 | Stress global/local decomposition |

---

## Budget Tiers

| Tier | `mbasis` | `abasis` | `steps` | `dim` | `total_refresh_budget` |
|------|----------|----------|---------|-------|------------------------|
| `Small` | 128 | 32 | 200 | 16 | 5.0 |
| `Medium` | 512 | 128 | 1,000 | 64 | 20.0 |
| `Large` | 2,048 | 512 | 2,000 | 128 | 50.0 |

---

## Usage

### Quick start (smoke test)

```bash
cd examples/evaluation
kernos-eval --datasets WineQuality --tiers Small --n_seeds 2 --output_dir results
kernos-analyze --results results/results.csv --output_dir results
```

Or via `python -m`:

```bash
python -m examples.evaluation.cli \
    --datasets WineQuality \
    --tiers Small \
    --n_seeds 2 \
    --output_dir results

python -m examples.evaluation.analyze_cli \
    --results results/results.csv \
    --output_dir results
```

### Full sweep (all datasets, all tiers, all ablations)

```bash
kernos-eval --datasets all --tiers all --n_seeds 5 --output_dir results
kernos-analyze --results results/results.csv --output_dir results
```

Skip ablations to halve runtime:

```bash
kernos-eval --datasets all --tiers all --n_seeds 5 --no_ablations --output_dir results
```

---

## Outputs

### Metrics (per dataset × model × tier)

- **RMSE / MAE / R²** — predictive quality.
- **Train time (s)** — wall-clock fit time.
- **Inference latency (s)** — per-batch predict time.
- **Peak memory (MB)** — max resident memory via `tracemalloc`.
- **Refresh count** — number of discrete refreshes triggered.
- **Condition proxy** — `κ(ΦᵀΦ + λI)`.

### Stability-over-Refits

- `Var(RMSE)` across repeated initializations.
- `Var(train_time)`.
- `Var(κ(ΦᵀΦ + λI))`.

### Visualisations

- `pareto_{dataset}.png` — RMSE vs train time per tier.
- `ablations_{dataset}_{tier}.png` — relative RMSE change per ablation.
- `results_table.tex` — LaTeX table of Kernos + baselines (no ablations).

---

## Models Evaluated

### Main Method

- **Kernos** — full refresh-aware hybrid kernel with continuous `R` optimization.

### Baselines

- **Ridge** — dense exact ridge in original feature space.
- **Nystrom** — static Nyström feature ridge (no refresh, no local corrective).
- **RFF** — Random Fourier Features ridge.

### Ablations

| Name | Disabled component |
|---|---|
| `K-NoRefresh` | Discrete refreshes |
| `K-NoHysteresis` | Hysteresis (`b_t = 1` permanently) |
| `K-NoCooldown` | Cooldown |
| `K-NoResidAnchors` | Residual-aware anchor sampling |
| `K-NoOrthog` | Local orthogonalization |
| `K-NoDivPenalty` | Diversity penalty (`γ_div = 0`) |
| `K-NoFreeze` | Calibration freezing |

---

## Reproducibility

- Fixed random seeds: `[42, 43, 44, 45, 46]` (default, configurable via `--n_seeds`).
- Hardware: CPU-only.
- Stopping criteria: `max_steps` with eval every `max_steps // 20`.

---

## Extending

To add a new dataset, edit the appropriate `examples/evaluation/datasets/sources/<origin>_sources.py`
module and register it in `examples/evaluation/datasets/registry.py`:

```python
DATASET_REGISTRY["MyDataset"] = load_my_dataset
```

Run with `--datasets MyDataset`.
