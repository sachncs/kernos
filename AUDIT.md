# Kernos production-readiness audit

Date: 2026-10-13

Scope: local repository, Python package, tests/tooling, Astro site source and
the deployed site at https://sachncs.github.io/kernos/.

Deployment status: the local site is redesigned and passes the built-artifact
gates below, but the deployed GitHub Pages URL is still serving the previous
artifact. A direct HTTP check on 2026-10-13 returned 200 with the prior
artifact (`Last-Modified: Fri, 09 Oct 2026 20:07:26 GMT`): it still contains
the old Python 3.12 claim, Google Fonts, legacy blue/violet styling, the
missing `y_test` definition, the `0.8923` hard-coded output, and the
pre-redesign benchmark copy. Publishing the local result requires an
authorized repository/Pages update and remains an external release step.

## Implemented since the baseline audit

- Exported `Buffer` from the package root and aligned package/site metadata to
  `0.2.0`.
- Fixed `partial_fit`/pickle runtime synchronization and added regression
  coverage for public imports, resumed streaming and non-finite inputs. A
  pickled estimator now reconstructs its loop and continues its existing
  bundle instead of silently starting over.
- Made estimator `set_params` transactional and added explicit Plan checks for
  numeric types, integer ranges, finite values and evaluation cadence.
- Made `eval_every` effective: validation/refresh evaluation now runs only on
  the configured cadence.
- Added typed component factories for embedders, solvers and drift metrics;
  `Loop` now consumes protocol objects from that composition boundary instead
  of selecting those strategies internally.
- Added an injectable `LoopComponents` dependency graph with a default builder;
  orchestration tests now prove the loop can receive a prebuilt runtime graph.
- Centralized basis and cache strategy selection in typed registries, removing
  the remaining `Loop`/`Refresh` string branches for Nyström/Greedy and
  Full/Stream/Adaptive construction; registry behavior has direct unit tests.
- Enforced copy-on-write ownership and read-only NumPy arrays at the
  `Continuous`, `Discrete` and `Bundle` snapshot boundaries, with mutation
  regression tests.
- Added generated numerical invariant tests with Hypothesis for rank-deficient
  and varying-shape ridge systems; Hypothesis is now part of the development
  dependency gate.
- Expanded the CI compatibility matrix and package classifiers from Python
  3.10–3.12 to 3.10–3.14, and aligned README, FAQ and site claims.
- Added an optional `memory_limit_mb` preflight guard for `fit` and
  `partial_fit`, with conservative mode-aware estimates and regression tests.
- Added immutable `TrainingDiagnostics` snapshots exposing completed steps,
  evaluations, refreshes, drift/gain, budget usage and the memory estimate;
  diagnostics survive pickle-based streaming continuation.
- Added structured solver-event diagnostics: Cholesky jitter retries,
  iterative non-convergence and PCG breakdowns are distinguishable, counted,
  and covered by numerical regression tests.
- Hardened prediction boundaries: `predict` now rejects non-finite features,
  and `score` validates both feature and target shapes/values before metric
  computation, preventing silent broadcasting and invalid scores.
- Added the public `MemoryLimitExceeded` exception for configured preflight
  budget refusals; it retains `MemoryError` compatibility and has hierarchy
  and estimator regression coverage.
- PCG breakdowns now raise `IllConditionedError` instead of returning a
  potentially invalid coefficient vector; ordinary iteration-budget
  non-convergence remains observable and non-fatal.
- Fixed `Policy.transition()` mutating frozen `Discrete` snapshots through
  `object.__setattr__`; transitions now use copy-on-write replacement and
  explicitly preserve the prior bundle.
- Hardened local RBF construction for valid zero-anchor configurations and
  explicit embedding-dimension mismatches, with regression tests instead of
  allowing an `argpartition(kth=-1)` crash.
- Fixed the browser-visible syntax highlighter corruption that rendered nested
  span markup as literal `class=class=...` text. Added a site smoke test and
  made it part of the Pages workflow.
- Added a built-artifact accessibility contract test for document language,
  heading structure, mobile navigation semantics, tab/panel relationships,
  focus-visible and reduced-motion CSS, legal links, and malformed markup.
- Measured the actual palette against the page background, replaced the
  failing `ink-500` text token (4.09:1) with `ink-400` (6.64:1), and added a
  WCAG-AA contrast assertion to the built-site smoke test.
- Added cooperative `cancel_check` hooks to `fit` and `partial_fit`, a public
  `TrainingCancelled` exception, and tests proving cancellation does not
  publish partial state.
- Added optional positive `timeout_seconds` controls with a typed
  `TrainingTimeout` exception, using the same safe step-boundary semantics.
- Reconfiguration now invalidates fitted runtime state after validated
  hyperparameter changes, preventing old bundles from being used with new
  estimator settings.
- Made `kernos/_version.py` the canonical package version source via dynamic
  setuptools metadata, and added `scripts/check-version.py` plus a CI gate for
  site/package release metadata synchronization.
- Added lazy adapters for the optional evaluation console scripts so the core
  install does not eagerly import datasets/matplotlib; missing extras now
  produce an actionable installation message.
- Added a committed `uv.lock`, documented `uv sync --locked`, and added a CI
  lock-consistency job for reproducible development dependency resolution.
- Built a local wheel and installed it into a fresh virtual environment; the
  clean install imported the public package and verified its version and
  `Buffer` export. CI now builds both wheel and sdist and repeats that smoke
  test.
- Repaired the live hero/API examples so `y_test` is defined.
- Added an accessible mobile navigation menu, keyboard-operable API tabs,
  focus-visible styling and reduced-motion handling.
- Replaced the generic blue/purple “K” assets with one adaptive-network logo
  mark and a warmer ink/ember/teal palette.
- Removed the unverified hard-coded R² result from the hero/API examples and
  replaced it with an explicit dataset/seed/training-budget caveat.
- Added a visible benchmark-methodology panel linking the evaluation protocol
  and documenting splits, seed stability, budget tiers, metrics and caveats.
- Added explicit scope text directly beneath the benchmark headline cards:
  Small tier, five controlled CPU seeds, cross-dataset aggregation and a
  hardware/dataset caveat are now visible without opening the protocol.
- Corrected an overclaim in the ablation section: the checked results show
  `K-NoRefresh` can outperform full Kernos on RMSE, so the page now presents
  refresh as a measured workload-dependent trade-off.
- Wired benchmark headline cards and per-dataset rankings to the tracked
  `examples/evaluation/results/summary.md` artifact; the Astro build now fails
  if required summary fields or ranking counts drift.
- Extended the built-site smoke test to verify every hash navigation link has a
  matching target and that Astro emits a usable 404 page.
- Extended the built-site smoke test to fail on stale benchmark output claims
- Formalized the remaining runtime seams as injectable protocols for sampling,
  fusion, whitening, refresh, outer-step, budgeting, orthogonalization and
  policy construction; strict type checking covers the composition graph.
- Hardened direct feature-component boundaries: scalers reject empty or
  non-finite blocks, orthogonalizers reject invalid dimensions/regularization,
  and projectors reject non-finite or incompatible matrices before linear
  algebra. Regression tests cover each failure mode.
  and to require the methodology caveat, protocol link, split and seed details.
- Corrected the install panel’s unsupported cross-platform testing claim to
  distinguish the verified Python CI matrix from platform-specific validation.
- Added a post-deploy Pages smoke check that fetches the public URL and rejects
  stale benchmark output, stale platform claims and missing methodology text.
- Added a keyboard skip link targeting the main landmark and removed the
  unsupported cross-platform operating-system claim from structured metadata.
- Performed a direct mobile browser smoke check: the navigation disclosure
  opens with section links, API tab selection swaps panels, and ArrowRight
  keyboard navigation reaches the Streaming panel. A full automated browser
  regression suite remains future work.
- Added the public API reference to the site footer and made the built-site
  smoke test require the link.
- Moved the hero and API Python snippets into marked blocks in
  `examples/benchmarks/site_snippets.py`; Astro imports those blocks at build
  time, and pytest compiles and executes every rendered block with bounded
  fixtures to prevent syntax and runtime drift.
- Reworked the hero copy to lead with the practical user outcome—large-batch
  and streaming regression without repeated basis rebuilds—before introducing
  the research terminology.
- Added a clean-wheel fit/predict/pickle/streaming smoke script and wired it
  into the package CI job, extending artifact validation beyond import checks.
- Added a weekly scheduled Python dependency audit with `pip-audit`, alongside
  the committed lockfile and existing npm audit gate.
- Promoted the tracked pre-commit hooks to an explicit CI job so contributor
  hooks and required repository checks cannot silently diverge.
- Ran the exact `pre-commit run --all-files` command locally; all hooks pass
  after normalizing missing EOF newlines in legacy tracked files.
- Added a single public API reference covering supported exports, lifecycle,
  diagnostics, extension protocols and the semi-private naming convention.
- Rebuilt the current wheel locally, installed it into a fresh virtual
  environment, and passed the fit/predict/pickle/streaming smoke test at
  version `0.2.0`.
- Earlier baseline verification ran the full suite in an isolated Python 3.13
  environment (**245 passed in 36.55s**) and the repository Python 3.14
  environment; the expanded matrix evidence is recorded below.
- Re-ran the complete **257-test** suite in isolated local environments for
  Python 3.10, 3.11, 3.12 and 3.13, then restored the repository `.venv` on
  Python 3.14.8 and verified all 257 tests there as well. The declared Python
  interpreter matrix is now locally evidenced; dependency lower/upper-bound
  combinations remain a separate matrix gap.
- Exercised the declared lower dependency bounds in an isolated Python 3.10.21
  environment with NumPy 1.24.4, SciPy 1.10.1, scikit-learn 1.3.2,
  datasets 2.14.7 and matplotlib 3.7.5: the earlier 222 core unit/numeric/
  integration tests and 28 examples/site-snippet tests passed. The current
  lower stack runs 227 core unit/numeric/integration tests successfully, but is
  much slower (the full core run took 204.3 seconds, including the
  default-hyperparameter integration case),
  so its performance budget remains an explicit follow-up.
- Changed the public GridSearchCV snippet from unbounded `n_jobs=-1` to
  deterministic `n_jobs=1`; the lower-bound stack showed the former could
  oversubscribe a laptop during example validation.
- The current Python 3.14 run passes **259 tests**, including synchronized
  site-snippet, injected-component and detailed PCG outcome coverage.
- Added a repository-level `Makefile` with one reproducible `make verify`
  command for Python lint/format/type/API/test gates and site build,
  accessibility, browser and dependency-audit gates; its default interpreter
  is the local `.venv` with an explicit override available.
- Added `make coverage-check` and included it in `make verify`, enforcing the
  same 80% minimum coverage threshold locally that CI enforces; the current
  suite measures 94.99% coverage across 2,216 Kernos statements.
- Added `build`, `pre-commit` and `pip-audit` to the locked development extra;
  the local `make package` target now builds both sdist and wheel successfully.
- Updated package metadata to SPDX `MIT`/`license-files` format so release
  builds complete without the deprecated license metadata warnings.
- Re-ran the expanded `make verify` gate after adding the package build:
  Ruff, formatting, strict mypy, API/config gates, **257 Python tests**,
  sdist/wheel build, Astro build, accessibility/browser smoke and npm audit
  all pass locally.
- Added `make package-smoke` to the local gate; a fresh virtual environment
  installs the wheel, runs fit/predict/pickle/streaming continuation, and
  verifies optional evaluation CLI help works without extras while actual
  execution reports the missing extra clearly.
- Deferred optional evaluation and plotting imports until after CLI argument
  parsing, so `kernos-eval --help` and `kernos-analyze --help` work from a
  minimal wheel without `datasets` or `matplotlib`; fresh-wheel smoke coverage
  verifies both help and actionable failure behavior.
- Made `examples.evaluation` itself lazily expose `DATASET_REGISTRY` and
  extended the adapter boundary to normalize execution-time missing extras as
  well as import-time failures; `make package-smoke` now passes both CLI help
  and failure paths from a clean wheel.
- Added a regression test proving the lazy package boundary preserves the
  documented `from examples.evaluation import DATASET_REGISTRY` export.
- Added an immutable `SolverRecovery` record with typed outcomes, stage,
  retries, iteration count, condition number and exception-failure capture;
  initialization and refresh solver failures now publish diagnostics before
  re-raising.
- Corrected iterative-solver telemetry: PCG now has a backward-compatible
  detailed path that reports the actual convergence/breakdown iteration rather
  than incorrectly reporting the configured maximum after successful solves.
- Fixed Pages clean-install portability by pinning the workflow to Node 22 and
  declaring Astro's Vite peer `esbuild` explicitly; a fresh `npm ci` now
  resolves the same lockfile on the local and GitHub-hosted npm toolchains.
- Applied the semi-private naming convention to `Loop` orchestration services
  and refresh-reference state (`_solver`, `_refresh_pipe`, `_r_ref`, etc.),
  keeping estimator constructor parameters and fitted sklearn attributes public.
- Extended that convention to the `Refresh`, `Policy` and `Outerstep`
  orchestration objects (`_plan`, `_solver`, `_fuse`, `_drift`, etc.), so
  dependency graphs are not accidentally treated as public extension points.
- Corrected the design documentation so it describes the implemented
  `typing.Protocol` contracts, injectable `LoopComponents` graph and factory-
  based polymorphism instead of the obsolete duck-typing-only architecture.
- Added a README configuration synchronization gate that compares every
  documented constructor default with the live `Kernos` signature and runs as
  part of `make api-check` and CI lint checks.
- Added `scripts/resource-benchmark.py` and `make resource-benchmark`, a fresh
  deterministic workload that reports clean fit/predict time, forced-refresh
  fit time, high-water RSS delta, estimated allocation, runtime versions and
  platform metadata, with explicit configurable budgets for local regression
  checks.
- Added a CI lower-bound compatibility job that installs Python 3.10 with
  NumPy 1.24.4, SciPy 1.10.1 and scikit-learn 1.3.2 explicitly, then runs the
  core, numeric and integration suites without the locked modern stack.
- Added a separate latest-dependency CI job on Python 3.14 that intentionally
  bypasses `uv.lock` and exercises the current NumPy/SciPy/scikit-learn
  releases, preserving the lockfile as the reproducible release path while
  detecting upstream compatibility breakage.
- Added a macOS-host CI job using the locked Python 3.14 environment and core
  suites, complementing the Linux matrix with a second operating-system
  signal for native numerical behavior.
- Added a Windows-host CI job using the locked Python 3.14 environment and
  core suites, completing the declared three-host compatibility smoke matrix.
- Added a Linux/macOS/Windows resource-matrix CI job that uploads the deterministic
  clean/refresh benchmark JSON per host, providing comparable evidence for
  timing, RSS and estimated allocation without enforcing identical host
  timings.
- Added a stdlib-only benchmark aggregator and CI summary job that computes
  min/mean/max metrics across the host artifacts and uploads one reviewable
  summary JSON.
- Made the site browser smoke harness engine-selectable and added a Pages CI
  browser matrix for Chromium, Firefox and WebKit; local runs continue to use
  installed Chromium by default.
- Added reviewed desktop and phone PNG baselines under
  `site/visual-baselines/`, a pixel-diff harness using `pixelmatch`, and the
  explicit `make site-visual` local gate. The baseline is intentionally kept
  separate from cross-OS CI because font rasterization differs by host.
- Installed the local Playwright Firefox and WebKit engines and ran the full
  browser smoke contract successfully in Chromium, Firefox and WebKit.
- Corrected the pull-request checklist to use Ruff consistently and declared
  read-only repository permissions for the CI workflow.
- Added `twine check` to both the package build and CI artifact gates, and
  removed the remaining stale universal scikit-learn/drop-in claims from
  README, design docs, metadata and site feature copy.
- `make package` now passes `twine check` for both the wheel and sdist;
  rebuilt site metadata and feature copy pass the accessibility and browser
  smoke suites.
- Added a tag-controlled PyPI release workflow using trusted publishing,
  canonical tag/version validation, artifact metadata checks, and a protected
  `pypi` environment; local verification intentionally stops before publish.
- Migrated Python CI jobs from unconstrained `pip install` commands to
  `uv sync --locked --extra dev` plus `uv run`, so lint, typing, tests, audits
  and package builds now consume the committed dependency lock across the
  supported interpreter matrix.
- Updated the tag release workflow to use the same locked environment and
  clarified that production publishing is workflow-controlled rather than a
  manual developer-workstation upload.
- Replaced direct iterative-solver stderr writes with package logging and made
  the benchmark baseline hierarchy an enforced abstract base class.
- Activated the configured direct-solver condition-number guard and exposed
  the measured value through immutable training diagnostics; temporary
  finite-difference solves skip the expensive duplicate check by design.
- Completed the typed configuration views with the `stab_lmin` stability
  field and added a CI/local gate proving all 46 `Plan` fields are covered by
  exactly the grouped configuration boundary.
- Audited the estimator against scikit-learn's generic checker and removed
  the unsupported universal “drop-in” wording; the documented interface now
  states the explicit landmark/validation capacity contract.
- Removed remaining legacy blue/cyan/violet tokens from the site components,
  added explicit teal/ember design tokens, and verified the rebuilt page still
  passes contrast, accessibility and browser interaction checks.
- Executed every shipped synthetic benchmark demo in pytest with isolated
  temporary output directories; this caught and fixed an invalid GridSearchCV
  parameter grid that violated the estimator's `abasis <= 0.25 * mbasis`
  capacity contract.
- Expanded the Playwright smoke path to cover 320px layout containment,
  keyboard skip-link activation, mobile menu open/close, tab ARIA state and
  left/right wraparound. This caught and fixed a real API-card min-content
  overflow at small-phone widths by adding an explicit `min-w-0` boundary.
- Removed the remaining local site overclaim (“exactly like an sklearn
  estimator”) and stale blue semantic label; the accessibility smoke test now
  guards the scikit-learn-style wording and palette vocabulary.
- Aligned the Ruff version in pre-commit, the development dependency range and
  `uv.lock`, eliminating formatter drift between local hooks and CI.
- Replaced duplicated free-form solver event strings with the public,
  string-compatible `SolverEvent` enum; direct, Woodbury and iterative paths
  now emit the same controlled vocabulary.
- Added the public `SolverStage` enum and immutable
  `diagnostics_.last_solver_stage` so recorded solver events identify whether
  they came from initialization or refresh; the API reference and export gate
  cover the new diagnostic field, with integration coverage forcing and
  verifying a refresh-stage solve.
- Corrected the remaining product-code `setparams` naming violation to
  `set_params` in both embedders, with focused regression coverage; the
  Google-style Ruff docstring/name gates remain green.
- Documented the narrow Ruff naming exception for conventional uppercase
  matrix/vector notation (`X`, `U`, `R`, `S`) so the semi-private/public
  naming contract is explicit rather than hidden in lint configuration.
- Corrected solver diagnostic lifecycle so jitter and event fields reset for
  every solve and clean Cholesky paths report no fallback jitter; regression
  coverage verifies the current-call semantics.
- Added descriptive immutable `Plan` aliases for cryptic legacy hyperparameters
  and migrated core orchestration to those aliases without changing the flat
  sklearn-compatible constructor.
- Threaded cooperative cancellation/timeout checks through initialization,
  finite-difference sweeps, refresh scoring and solver boundaries while
  retaining atomic fit publication; integration coverage continues to verify
  prior snapshots survive interruption.
- Hardened the public sampling and basis constructors/forward paths against
  empty, non-finite and dimensionally inconsistent inputs, with explicit
  regression coverage for Nyström, Greedy and random-feature bases.
- Corrected evaluation documentation and benchmark copy so `tracemalloc`
  results are labeled peak Python allocation rather than max resident memory;
  native NumPy/RSS measurement remains an explicit open item.
- Exported `SolverEvent` from both `kernos` and `kernos.core`, with a public
  import regression test and API-reference coverage.
- Added a pinned Playwright Chromium smoke test covering desktop navigation,
  mobile menu disclosure, API-tab keyboard traversal and reduced-motion context;
  Pages CI installs Chromium and runs the same built-preview test.
- Added a bounded native-RSS/runtime integration fixture for a representative
  small fit on Unix runners; it records elapsed time, high-water RSS delta and
  the model’s preflight estimate without claiming cross-host determinism.
- Added `scripts/check-api-reference.py` and a CI gate that verifies the
  documented compatibility names still exist in the actual public exports.
- Hardened all cache modes against malformed batches: target rank, sample
  count, feature width, finiteness and positive configuration values are now
  checked before any cache mutation.
- Hardened direct sampler, RBF, gate and fusion boundaries against invalid
  mixtures, neighbor counts, gate ranges, scaling factors, dimensions and
  non-finite arrays, with focused regression coverage.
- Hardened direct solver and preconditioner constructors against invalid ridge,
  jitter, retry, factor, cap, tolerance, iteration and epsilon settings so
  extension users receive fail-fast errors outside the `Plan` path.
- Hardened lower-level numeric utilities (`condition`, jittered Cholesky and
  PCG) against malformed matrices and invalid control parameters, with direct
  tests for symmetry, shape, finiteness and convergence controls.
- Tightened stability configuration invariants for `Plan`, `Whitening` and
  `Scaler`, including positive spectral floors, jitter bounds, symmetric kernel
  matrices and finite landmark inputs.
- Cleaned all Ruff check/format failures in the shipped Python tree.
- Made strict mypy pass for all 62 source files and added typed structural
  protocols for feature maps, solvers, residual solvers, orthogonalizers and
  drift metrics.
- Added [`docs/engineering-principles.md`](docs/engineering-principles.md) with
  the public/private naming convention, Google-style documentation rules,
  lifecycle, polymorphism, numerical resilience and testing gates.
- Added an explicit `Lifecycle` enum and atomic fit staging: cancelled or
  timed-out refits preserve the prior fitted snapshot, while successful
  `partial_fit` transitions the estimator into streaming state.
- Extended immutable training diagnostics with the active memory mode and
  elapsed wall-clock time, with integration coverage for both fields.
- Added immutable typed configuration views for embedding, basis, solver,
  training, refresh, ablation and validation concerns while preserving the
  sklearn-compatible flat constructor.
- Added deterministic integration coverage for bounded training steps,
  evaluation cadence and refresh-budget consumption; timing thresholds remain
  intentionally outside CI until benchmark fixtures are standardized.
- Added a full-vs-stream integration contract: identical seeded training must
  produce prediction parity, while the stream mode reports a lower preflight
  memory estimate for the same workload.
- Centralized solver input validation for all built-in solvers, rejecting bad
  dimensions, sample-count mismatches and non-finite values with the public
  `ShapeError` taxonomy before matrix operations begin.
- Added solver convergence status to immutable training diagnostics and a
  non-convergence regression test for the iterative solver.
- Added predictor-side shape and finiteness validation for weights, covariance
  matrices and query features, with regression coverage for width and covariance
  mismatches.
- Added direct component validation for scaler, projector and orthogonalizer
  inputs so empty, non-finite and incompatible blocks fail with actionable
  errors before numerical work.
- Migrated the site from Astro 4 / the deprecated Tailwind integration to
  Astro 7.3.8 with Tailwind 4 via `@tailwindcss/vite`; the site now builds and
  `npm audit --audit-level=high` reports zero vulnerabilities.
- Enabled Ruff's Google pydocstyle family for the library and completed the
  constructor/method docstrings required by that gate. Tests and benchmark
  harnesses remain exempt from product API docstring rules by design.

## Verification snapshot

- Python test suite: **259 passed**.
- Python coverage baseline: **95.00%** across 2,219 statements against the
  configured 80% gate; `make verify` enforces this threshold locally.
- `ruff check kernos tests examples`: passed.
- `ruff format --check kernos tests examples scripts`: passed; 138 files already
  formatted.
- `mypy kernos`: passed with no issues in **63 source files**.
- `pip-audit`: passed locally with no known vulnerabilities; the editable
  local `kernos` project is expectedly skipped because it is not indexed on
  PyPI.
- `npm run build` in `site/`: passed with Astro 7.3.8.
- `npm audit --audit-level=high`: passed with **0 vulnerabilities**.
- `npm run test:browser` in `site/`: passed against the built preview with
  Chromium 156, covering 320px/390px responsive navigation, keyboard
  skip-link and tabs, ARIA state transitions, layout containment and reduced
  motion.
- Local browser smoke: rebuilt page exposes the outcome-led hero; at a 390×844
  viewport the mobile menu opens and the GridSearchCV tab selects its panel.
- Local verification now covers Python 3.10.21 through 3.14.8; the lower-bound
  dependency stack is separately exercised and documented as materially slower.
- `make verify`: passed end to end on the local `.venv`.

Passing tests now cover the repaired public import, streaming update, pickle
resume, finite-input paths, every shipped synthetic benchmark demo and the
clean wheel path. Local checks include native-RSS workload and reviewed
desktop/phone pixel baselines; CI collects resource artifacts across Linux,
macOS and Windows and runs cross-browser engine coverage. Cross-OS pixel
normalization and statistical trend aggregation remain open.
The wheel/sdist build and clean wheel import are now covered locally and in
CI.

## Issue inventory

### P0 — externally visible correctness and release blockers

The original P0 findings (1–7) are addressed in this working tree. They are
retained below as the traceable baseline; the remaining release blockers are
the packaging, compatibility, example-execution and deployment gates in the
P1/P2 backlog.

1. **Public API export mismatch.** The docs and site show `from kernos import
   Kernos, Buffer`, but `Buffer` is defined in `kernos.core.plan` and is not
   exported from `kernos.__init__`. The advertised snippet fails immediately.
2. **Live API example is not executable.** `site/src/components/Api.astro`
   and the deployed page call `model.score(X_test, y_test)` without defining
   `y_test`.
3. **Version drift.** `pyproject.toml`, `kernos/_version.py`, issue templates,
   and deployment docs say `0.1.0`; the deployed site, changelog and install
   transcript say `0.2.0`. The changelog also lists future GPU/learned-embedding
   work that is not represented in the package metadata.
4. **`partial_fit` leaves prediction caches stale.** It replaces `bundle_` but
   does not rebuild `predictor_` or `basis_`, while `predict()` uses the cached
   predictor. A successful update can therefore produce predictions using old
   weights.
5. **Unpickled models cannot safely continue streaming.** `__getstate__`
   deliberately removes `loop_`, but `partial_fit()` assumes `self.loop_` is
   usable whenever the attribute exists. After unpickling, it can call
   `.step()` on `None`.
6. **Site license link is broken.** The footer uses `href="LICENSE"`, but the
   static site does not publish `LICENSE` into `site/public` or `dist`.
7. **Security debt in the web toolchain.** The current Astro 4/Tailwind 3
   dependency tree reports a critical Astro advisory plus high-severity
   transitive issues. The site is static today, but the vulnerable toolchain
   still runs in CI/development and must be upgraded or explicitly risk-
   accepted.

### P1 — reliability, maintainability and architecture

Findings 8–18, 21–29, 45 and 47 have been addressed to varying degrees by
the current type, validation, protocol, numerical and documentation changes.
They remain listed because the deeper redesign and automated gates are not
finished. Findings 12, 17, 18, 19, 22, 24, 27, 28 and 29 remain open or
only partially covered; the implemented list above is the source of truth for
what has already landed.

8. **Type-checking is addressed.** Strict mypy passes for all 62 source files;
   third-party sklearn typing remains isolated behind the estimator boundary.
9. **Formatter contract is addressed.** Ruff formatting/checking now covers
    `kernos/`, `tests/`, `examples/` and `scripts/`; pre-commit and CI run the
    same Ruff formatter, and contributor docs no longer name Black.
10. **CI coverage is broad but split across workflows.** Ruff covers shipped
    Python trees, console scripts have import/smoke coverage, and package/site
    builds have dedicated gates; cross-artifact orchestration remains a process
    gap.
11. **Python dependency policy is now established.** `uv.lock` is committed,
    CI checks lock consistency, and a weekly `pip-audit` job scans installed
    package dependencies. Lower-bound declarations remain intentionally broad,
    so dependency update review is still required when the lock changes.
12. **Supported-version matrix is addressed locally and in CI for the lower
    bound.** The full 257-test suite passes on Python 3.10.21, 3.11.16, 3.12.14,
    3.13.16 and 3.14.8 in isolated/local environments. The explicit lower-bound
    CI job now covers the declared minimum NumPy/SciPy/scikit-learn stack;
    latest-dependency CI now exercises the unconstrained upper side and
    macOS/Windows host jobs cover both non-Linux platforms.
13. **Input validation is addressed at estimator, solver and feature-component
    boundaries.** Finite values, shapes, sample counts, feature drift,
    hyperparameter ranges, memory preflight, scaling, projection and
    orthogonalization inputs are covered before numerical work; overflow
    behavior for extreme user scales remains a numerical test opportunity.
14. **Transactional configuration is addressed.** `set_params` validates a
    complete candidate before mutation and invalidates fitted state only after
    successful reconfiguration.
15. **Lifecycle modeling is addressed.** `Lifecycle` and atomic fit staging
    cover unfitted, fitting, fitted and streaming transitions, including pickle
    resume and cancellation preservation.
16. **Snapshot immutability is addressed.** Array-backed state copies and
    freezes NumPy arrays at `Continuous`, `Discrete` and `Bundle` boundaries.
17. **Dependency inversion is addressed at the loop boundary.** `Loop` accepts
    an injectable `LoopComponents` graph, including policy construction,
    sampling, fusion, orthogonalization, refresh, outer-step and budget seams;
    the default builder keeps the convenient plan-driven path.
18. **Polymorphism is addressed at the algorithm seams.** Typed protocols and
    registries cover embedders, solvers, drift metrics, basis strategies, cache
    modes, sampling, fusion, whitening, refresh, outer-step, budgeting,
    orthogonalization and refresh policy. Implementations remain replaceable
    without adding string conditionals to the loop.
19. **Configuration remains a compatibility-shaped mega-object.** One large
    `Plan` mixes
    model, optimizer, numerical, refresh, ablation and validation concerns;
    many related fields have cryptic names (`wr`, `worth`, `wdiv`, `fdeps`,
    `cool`, `warm`). Typed immutable configuration views now group those
    concerns without breaking the flat sklearn API. Descriptive read-only
    aliases now drive core orchestration, but the underlying constructor and
    legacy fields remain flat until a future major API revision.
20. **Public surface is now documented.** Supported exports, lifecycle,
    diagnostics, extension protocols and visibility rules are centralized in
    `docs/api-reference.md`; generated API pages and compatibility tooling are
    still future improvements. The estimator intentionally exposes a
    scikit-learn-style API rather than claiming universal compliance with
    `sklearn.check_estimator`: its landmark/validation capacity contract is
    explicit and documented.
21. **`eval_every` is addressed.** Validation/refresh evaluation is gated by
    the configured cadence, with deterministic integration coverage proving the
    number of evaluations for a fixed step budget.
22. **Numerical failure policy is substantially addressed.** Explicit inverses
    were removed, solver input failures use `ShapeError`, all built-in solvers
    emit the same `SolverEvent` vocabulary, and immutable `SolverRecovery`
    records expose stage, jitter/retries, conditioning and final outcome.
   Actual PCG iteration counts are now captured; threshold-specific payloads
   and richer breakdown diagnostics remain future refinements.
23. **Memory controls are substantially enforced.** Conservative estimates,
    `memory_limit_mb`, cancellation, native-RSS smoke coverage and the
    deterministic clean/refresh resource benchmark now exist. Broad
    near-budget trend collection across hosts remains open.
24. **Evaluation isolation has an explicit compatibility boundary.** Core
    imports and CLI help do not eagerly load datasets/matplotlib, while actual
    evaluation gives an actionable missing-extra error. The `examples*`
    modules are intentionally included in the distribution because the
    declared `kernos-eval`/`kernos-analyze` entry points import them; excluding
    them would make the `examples` extra unusable. A separately versioned
    evaluation distribution remains an optional future major-version split,
    not an unresolved behavior in the current release.
25. **Packaging smoke coverage is addressed.** CI builds wheel/sdist, performs
    clean imports, checks CLI optional-dependency behavior and runs fit/predict/
    pickle/streaming smoke coverage from a fresh wheel.
26. **Property/fuzz coverage is addressed for core invariants.** Hypothesis,
    adversarial solver validation, cancellation, memory, lifecycle and mode
    integration tests supplement fixed examples; broader numerical fuzzing
    remains a future expansion.
27. **Performance/regression budgets are substantially addressed.** Deterministic
    integration guards bound training steps, evaluation cadence, refresh-budget
    consumption and full-vs-stream parity; the resource benchmark bounds clean
    fit, forced-refresh fit, prediction, RSS and estimated allocation. Broad
    workload trends are now collected as Linux/macOS/Windows artifacts using
    native RSS measurement on each host and aggregated into min/mean/max CI
    summary JSON. Historical trend retention and alerting remain future work.
28. **Error taxonomy is substantially adopted.** Built-in solver input failures
    use `ShapeError` before numerical work, estimator validation intentionally
    exposes sklearn-compatible `ValueError`, and solver event/recovery codes
    are typed. Richer numeric-breakdown payloads remain future work.
29. **Documentation synchronization is substantially addressed.** The API
    export/name gate, grouped configuration-view gate and README constructor
    default synchronization gate run locally and in CI. Fully generated API
    pages and mathematical-claim checks remain future work.

### P1 — website UX, accessibility and content

Findings 30–34, 36, 38 and 39–41 are addressed or covered by local/browser
smoke checks in the working tree. Findings 35 and 37 remain open or only
partially addressed; functional browser regression now runs across three
engines, while visual and claim-uncertainty refinements remain explicit.

30. **Mobile navigation was incomplete.** It is now implemented with a native
    responsive menu and five verified section links.
31. **Tabs were only visually accessible.** They now expose stable ARIA tab/
    panel relationships, selection state and keyboard arrow behavior.
32. **Motion preferences were ignored.** Reduced-motion CSS and browser smoke
    coverage now disable the reveal/pulse motion path.
33. **Focus treatment was not designed.** A site-wide `:focus-visible` system,
    skip link and keyboard smoke coverage now protect keyboard navigation.
34. **The original palette had weak contrast.** The ink/ember/teal palette now
    has measured contrast tokens and accessibility smoke coverage; per-host
    visual rasterization is tracked separately.
35. **The page was too long and repetitive.** The capability, architecture and
    evidence sections now use left-anchored editorial headings with section
    rules rather than repeating the same centered pill/gradient treatment.
    The API and install surfaces retain distinct split-panel conversion layouts.
36. **The hero previously led with research jargon.** The working tree now
    leads with the practical outcome—large-batch and streaming regression
    without repeated basis rebuilds—and follows with the technical description.
37. **Claims have context but lack per-card uncertainty.** The public quickstart no longer presents an
    unverified hard-coded score and now states that output depends on the
    dataset, seed and training budget. The benchmark section now links and
    summarizes the evaluation protocol, and the headline cards now state the
    Small tier, five-seed aggregation and hardware/dataset caveat. Per-card
    confidence/error bars and runtime context remain future improvements. The
    ablation narrative is now explicitly consistent with the recorded
    `K-NoRefresh` result rather than claiming refresh universally improves
    accuracy.
38. **Code examples are addressed.** The hero and API tabs import marked
    blocks from `examples/benchmarks/site_snippets.py`; each block is now
    self-contained, syntax-checked and executed by pytest with bounded demo
    data.
39. **External font loading was a resilience/privacy dependency.** The working
    tree now uses local/system font stacks and has no Google Fonts runtime
    dependency.
40. **Interaction/state QA is substantially covered.** Built-artifact coverage
    verifies hash-link targets and 404 output, and the Playwright smoke test
    covers desktop/mobile navigation, API-tab keyboard flow and reduced-motion
    context. Pages CI runs that functional contract in Chromium, Firefox and
    WebKit; reviewed desktop/phone pixel baselines are also checked locally,
    while cross-OS pixel normalization remains absent.
41. **Brand assets were inconsistent.** The working tree now uses one adaptive
    network mark across `Logo.astro`, `public/logo.svg`,
    `public/logo-mark.svg`, and `favicon.svg`, with light/dark-safe semantics.

### P2 — process and product quality

Finding 42 is addressed by `docs/engineering-principles.md`; findings 43–47
are now either partially addressed or explicitly marked with their remaining
gaps below.

42. **No explicit engineering principles document.** Existing docs mention
    immutability and duck typing but do not define boundaries, ownership,
    compatibility policy, observability, review gates, or failure semantics.
43. **Contributor tooling is aligned.** A tracked pre-commit configuration now
    covers whitespace, YAML, Ruff and formatting, CI runs it explicitly, and
    contributor/deployment docs now identify Ruff as the sole formatter.
44. **Release/deployment automation is partially complete.** Version metadata
    is canonicalized, CI builds wheel/sdist artifacts, and Pages has build,
    deploy and post-deploy verification. PyPI publishing remains a reviewed,
    secret-gated manual/tag workflow and is documented as such.
45. **Version automation is partially complete.** A canonical version source,
   package/site synchronization check, and changelog-heading check now run in
   the local and CI gates. Changelog content quality, release notes, and
   release/tag creation still require manual edits and review by design.
46. **Observability is addressed for accepted and rejected streaming input.** Training now exposes immutable
    diagnostics for steps, refreshes, evaluations, drift, gain, budget,
    estimated memory, memory mode, elapsed time, conditioning, typed solver
    events, an immutable per-stage recovery record, accepted sample counts and
    rejected batch/sample counts. Invalid batches remain rejected transactionally;
    fit-time rejection before a model exists has no diagnostics object to update.
47. **Cancellation and timeout are cooperative only.** Checks now run at
    initialization, finite-difference, refresh-scoring and solver boundaries,
    but cancellation cannot interrupt a single native NumPy linear-algebra call
    already in progress.

## Remediation plan

### Completed foundation and redesign phases

The original baseline, correctness, architecture and first website-redesign
phases are implemented in this working tree. Evidence includes the canonical
version source, `Makefile`, locked CI, pre-commit, strict mypy, 259 tests,
95.00% coverage, wheel/sdist smoke tests, injectable protocol seams,
copy-on-write state, lifecycle/errors/diagnostics, tested site snippets,
  responsive navigation, keyboard tabs, reduced motion, contrast checks,
  screenshot baselines, a reusable live-site contract, stale-deploy markers and
  the unified SVG brand assets.
  The “Implemented since the baseline audit” section
above is the authoritative change log.

### Remaining production-hardening phases

1. **Compatibility matrix:** retain the Python 3.10–3.14, explicit lower-bound,
   latest-dependency and three-host jobs; review CI artifacts and investigate
   any platform-specific failures before declaring the matrix complete.
2. **Numerical failure contract:** the typed `SolverRecovery` record now
   exposes stage, retry count, jitter, condition number, convergence status and
   final decision through immutable diagnostics. Threshold-specific fields and
   richer breakdown payloads remain future improvements.
3. **Resource/performance evidence:** the deterministic local benchmark and
   Linux/macOS/Windows CI artifact matrix now
   reports native high-water RSS, clean and forced-refresh fit latency,
   prediction latency, estimated allocation and host/runtime metadata with
   configurable non-flaky budgets. Statistical aggregation and alerting over
   those artifacts remain future improvements.
4. **Package boundary:** current release behavior is deliberate and tested:
   evaluation code ships with the console entry points, while heavy runtime
   dependencies remain optional and lazy. If package size or release cadence
   requires it, split evaluation into a separately versioned distribution in a
   future major release.
5. **Documentation synchronization:** API exports, typed configuration views
   and README constructor defaults are now gated in CI. Fully generated
   API/default tables and a release checklist that updates changelog and
   versioned compatibility notes together remain future improvements.
6. **Observability and interruption:** accepted/rejected streaming counters and
   per-stage recovery metadata are implemented; retain cooperative cancellation
   semantics and document that a single native BLAS call cannot be preempted.
   Fit-time rejections are retained in private estimator counters and published
   on the next successful diagnostics snapshot; no partial model state is
   published during the failed attempt.
7. **Visual/browser QA:** functional browser coverage now runs successfully
   locally and in Pages CI across Chromium/Firefox/WebKit, and reviewed
   desktop/phone screenshot baselines are checked locally. Cross-OS pixel
   normalization remains open.
8. **Deployment:** the public URL currently returns HTTP 200 but still serves
   the older site artifact (including the prior Google Fonts dependency), not
   this verified working-tree redesign. The reusable `test:live` contract now
   passes against local preview output and is wired into Pages with retries.
   Publish the local site through the authorized Pages workflow, then record
   the deployed commit/version. This is the only item requiring external
   repository state rather than local code changes.

### Completion criteria

- Wheel/source installs, docs quickstart, all site snippets and all shipped
  synthetic demos execute successfully.
- CI and local `make verify` cover lint, format, strict typing, API/config
  gates, tests, coverage threshold, packaging, clean install, site build,
  browser/accessibility smoke and dependency audit policy.
- `fit`, `predict`, `partial_fit`, pickle restore and sklearn cloning have
  explicit regression coverage.
- All algorithm seams are typed and injectable, with documented semi-private
  naming and engineering principles.
- The site has one canonical brand asset source, tested keyboard/mobile/
  reduced-motion behavior, responsive code blocks, measured contrast and no
  broken internal links.
- The supported compatibility/resource/diagnostic evidence above is complete,
  and the redesigned artifact is verified after authorized deployment.
