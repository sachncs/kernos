# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-09-11

### Added

- `.github/CODEOWNERS` defaulting review assignment to `@sachncs`
- Component selectors on `Plan` and `Kernos`: `embedder` (linear/identity),
  `basis` (nystrom/greedy), `solver` (direct/iterative/woodbury),
  `drift` (frobenius/spectral) so the alternative implementations under
  `kernos.embed`, `kernos.basis`, `kernos.solver`, and `kernos.policy.drift`
  are reachable through the public API
- `Discrete.basis_wz` field to carry the on-landmarks kernel matrix used
  by the Greedy basis
- `Iterative.residual` method so the residual-aware anchor sampler works
  with the iterative solver
- CI coverage gate at 80% via `pytest --cov=kernos --cov-fail-under=80`
- CI `ruff format --check` job alongside `ruff check`
- Integration tests covering default hyperparameters and every `Buffer`
  memory mode
- `test_set_params_validates` test exercising the Plan re-validation on
  `set_params`

### Changed

- `Loop.maybe_refresh` now computes the actual validation RMSE drop and
  threads it into `Policy.decide` so `plan.gain * plan.rcost` is
  enforced; refreshes that fail the threshold are discarded
- `Loop.__init__` instantiates a `Budget(plan.budget)` and gates each
  refresh on `budget.can(rcost)` followed by `budget.spend(rcost)`
- `Kernos.fit`, `Kernos.predict`, and `Kernos.partial_fit` raise explicit
  `ValueError`s on 1-D / mismatched-shape / feature-count-drift inputs
  instead of letting the error surface from numpy
- `Kernos.fit` rejects datasets with fewer than `ceil(mbasis / (1 - val_frac)) + 2`
  rows up front with an actionable message
- `Kernos.set_params` builds a transient `Plan` so `Plan.__post_init__`
  invariants fire on configuration changes
- `Kernos.score` collapses an undefined R² (constant-y target) to `0.0`
  instead of silently propagating `nan`
- `Refresh.run` accumulates fused features through a `Cache` selected by
  `plan.mode` (Full / Stream / Adaptive), making the memory-mode
  parameter actually affect accumulation
- `Profile.refresh_times` records durations (deltas between consecutive
  refresh callbacks) instead of raw `time.perf_counter()` readings
- `.pre-commit-config.yaml` swapped `black` and a pinned `mypy` hook
  for `ruff format` so pre-commit matches the pyproject configuration
- `.github/PULL_REQUEST_TEMPLATE.md` test-path corrected from
  `tests/numerical/` to `tests/numeric/`
- `examples/benchmarks/README.md` rewritten as a proper Markdown
  document instead of a triple-quoted Python module

### Fixed

- `kernos/linalg.py` empty re-export module deleted (no importers
- `.github/FUNDING.yml` placeholder URLs stripped, leaving only the live
  GitHub sponsor entry
- `docs/architecture.md` and `docs/design.md` aligned with the actual
  public surface: `Continuous`/`Discrete`/`Bundle` (not the
  non-existent `ContinuousState`/`DiscreteState`/`FullState`),
  `dataclasses.replace` (not the fictional `copy_with`), duck-typed
  interfaces (not `typing.Protocol`), and the real `stab_*` fields on
  `Plan` (not the non-existent `NumericsConfig`)
- `docs/faq.md` parameter names corrected from pre-rebrand names
  (`memory_mode`, `lambda_reg`, `max_steps`, `m_g`, `m_l`, `embedding_dim`,
  `delta_hi`, `t_cool`, `disable_*`) to the current `mode`, `ridge`,
  `steps`, `mbasis`, `abasis`, `dim`, `drift_hi`, `cool`, `no*`

## [0.1.0] - 2026-01-01

Initial release under the pre-rebrand `aware-kernel` name (later
rebranded to `kernos`).  This entry documents the original public
surface; the equivalent functionality on the current rebrand is
covered in `[Unreleased]` and `[0.2.0]`.

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
