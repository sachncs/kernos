# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Comprehensive Google-style docstrings across all 34 source modules
- Enhanced module-level documentation explaining package responsibilities and architecture
- Algorithm descriptions, complexity analysis, and mathematical formulas in docstrings
- Thread-safety notes, design rationale, and recovery strategies in exception docstrings
- References to academic papers (Drineas & Mahoney 2005, Arthur & Vassilvitskii 2007, Spall 1992)
- MIT License file
- CONTRIBUTING.md with development guidelines
- CODE_OF_CONDUCT.md (Contributor Covenant v2.1)
- SECURITY.md with vulnerability reporting policy
- CHANGELOG.md following Keep a Changelog format
- .editorconfig for consistent formatting
- .gitattributes for line ending normalization
- GitHub issue templates (bug report, feature request)
- GitHub pull request template
- GitHub Actions CI workflow (lint, type-check, test)
- Dependabot configuration for automated dependency updates
- Funding configuration with placeholder URLs
- docs/getting-started.md with detailed setup instructions
- docs/architecture.md with full module documentation
- docs/deployment.md for publishing guidelines
- docs/faq.md with common questions and answers
- docs/evaluation-protocol.md specifying datasets, budget tiers, models, ablations, hyperparameter tuning, metrics, and reproducibility defaults for the evaluation suite
- `examples.evaluation` sub-package implementing the full real-world evaluation protocol as a proper Python package (datasets, models, runner, reporting submodules)
- `examples.benchmarks` sub-package with five runnable API demos (`quickstart`, `compare_baselines`, `memory_modes`, `ablation_walk`, `grid_search_demo`)
- `examples/README.md` top-level index for the examples tree
- `[project.scripts]` console-script entry points `kernos-eval` and `kernos-analyze` (replacing the old `python -m scripts.*` invocation)
- `examples` extras in `pyproject.toml` bundling HuggingFace `datasets` and `matplotlib`
- Smoke tests under `tests/examples/` covering registry contents, public API surface, CLI argument-parser build, and demo importability

### Changed

- Rebranded package from aware-kernel to kernos
- Renamed public estimator `AwareKernelEstimator` to `Kernos`
- Renamed hyperparameters to match the new `Plan` schema (`dim`, `mbasis`, `abasis`, `ridge`, `mode`, `steps`, …)
- Replaced `AblationConfig` dataclass with `Plan` boolean ablation flags (`noref`, `nohyst`, `nocool`, `noresid`, `noorth`, `nodiv`, `nofreeze`)
- Renamed baseline classes (`NystromRidgeBaseline` → `Nystrom`, `RandomFeatureBaseline` → `Random`) under `kernos.bench.baseline`
- Renamed metrics helper `compute_all_metrics` → `allmetrics` under `kernos.bench.metric`
- Renamed `MemoryMode` enum to `Buffer` under `kernos.core.plan`
- Replaced `TrainingLoop` with `Loop` under `kernos.loop.loop`
- Updated all documentation, GitHub metadata, and examples to use the new API
- Converted semi-private naming conventions to public API (removed `_` prefix from attributes with property accessors)
- Replaced `print` statements with `logging` module in training callbacks
- Updated tests to reflect public attribute naming
- Updated README.md with badges, detailed sections, and improved examples
- Updated pyproject.toml with correct repository URLs, classifiers, and metadata
- Updated .gitignore with additional Python-specific patterns
- Rebranded `examples/eval_real_world/` → `examples/evaluation/` and split the 519-line `run_evaluation.py` monolith into focused submodules (`datasets/{registry,sources/{sklearn,openml,huggingface,synthetic}}`, `models/{tiers,factories,ablations}`, `runner/{splits,single,diagnostics,suite}`, `reporting/{result_record,markdown,csv_writer,plots,latex}`)
- Renamed `data_loaders/` → `datasets/` and `DATASET_LOADERS` → `DATASET_REGISTRY`; relocated loaders into `datasets/sources/<origin>_sources.py`
- Replaced `scripts/run_evaluation.py` with `examples/evaluation/cli.py` and `scripts/analyze_results.py` with `examples/evaluation/analyze_cli.py`
- Replaced `sys.path.insert` hacks in the example scripts with proper install-based imports (`pip install -e ".[dev,examples]"`)
- Expanded `setuptools.packages.find` to include `examples*` alongside `kernos*`
- Restructured `README.md` for audience orientation, plain-language configuration, and clearer navigation, modelled on the `sachncs/convexfolio` README design
- Updated project tree in `README.md` to reflect the new `examples/evaluation/` and `examples/benchmarks/` layout
- Added a "Your first run — CLI" section showcasing the new `kernos-eval` entry point
- Updated `docs/getting-started.md` cross-references to the new examples layout

### Removed

- `examples/eval_real_world/` directory (replaced by `examples/evaluation/`)
- Stale artifacts in `examples/eval_real_world/results/` (regenerated by the new suite)
- Broken reference to a non-existent `EVALUATION_PROTOCOL.md` (replaced with `docs/evaluation-protocol.md`)
- `sys.path.insert` hacks in the example scripts

## [0.1.0] - 2026-01-01

### Added

- Initial release of aware-kernel
- `AwareKernelEstimator` sklearn-compatible API
- Refresh-aware hybrid continuous-discrete low-rank kernel learning
- Global Nyström basis with soft-truncated whitening
- Local corrective features with residual-aware anchor sampling
- Residual orthogonalization (local in global nullspace)
- Feature calibration and fusion (logistic gate)
- Refresh controller (drift-aware with cooldown, warmup, hysteresis, budget)
- Two memory modes: cached O(nm) and streamed O(m^2)
- Ridge regression solver (direct Cholesky and iterative PCG)
- Numerical stabilization (eigenvalue clipping, soft truncation, Cholesky with jitter)
- Training loop with objectives, callbacks, and outer-loop optimizer
- Inference with mean and variance prediction
- Evaluation framework with synthetic datasets, baselines, and metrics
- Comprehensive test suite (unit, numerical, integration) with ~91% coverage
- Design document (docs/design.md)
- Real-world evaluation examples (examples/eval_real_world/)
