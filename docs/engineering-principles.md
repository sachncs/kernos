# Engineering principles

This is the working contract for production changes to Kernos. Each principle
should be observable in code review, tests, or runtime diagnostics.

## Naming and visibility

- Names without a leading underscore are public and may be documented, imported,
  and supported.
- A single leading underscore marks a module-private helper or implementation
  detail. It may change without a compatibility promise.
- Double-leading-underscore name mangling is reserved for genuine collision
  avoidance, not general privacy.
- Names ending in `_` follow scikit-learn's fitted-attribute convention
  (`bundle_`, `n_features_in_`, `predictor_`). They are created only after
  successful fitting and replaced atomically.
- Constants use `UPPER_SNAKE_CASE`; types use `PascalCase`; functions,
  parameters, and locals use `snake_case`.
- Mathematical matrix/vector notation may retain conventional uppercase names
  such as `X`, `U`, `R`, and `S` at numerical boundaries; this is the narrow
  exception represented by the Ruff naming configuration and does not permit
  camelCase, compressed method names, or undocumented public state.

## Interfaces and polymorphism

- Swappable components implement a protocol in `kernos.core.interfaces`.
- Callers depend on the smallest behavior they need, not on a concrete class.
- Strategy selection belongs in a factory/registry boundary; algorithmic code
  must not grow string-condition branches for every new implementation.
- Composition is preferred for model assembly. Inheritance is used only where
  substitutability is real, such as baseline and cache strategies.

## State and lifecycle

- `Plan` is immutable configuration. `Bundle` is an immutable training snapshot;
  transitions return a replacement snapshot.
- NumPy arrays inside snapshots are owned by the snapshot boundary and must not
  be mutated in place after publication.
- Estimator lifecycle is explicit and represented by `Lifecycle`: unfitted ->
  fitting -> fitted/streaming. Training is staged locally and publishes a new
  fitted snapshot only after successful completion; cancelled or timed-out
  refits preserve the previous snapshot.
- Derived runtime objects are rebuilt from the authoritative bundle after fit,
  partial fit, refresh, and deserialization.

## Numerical resilience

- Validate shape, finiteness, dtype, and minimum sample requirements before
  numerical work.
- Prefer factorizations and linear solves over explicit matrix inversion.
- Every fallback has a bounded retry policy and produces an actionable error or
  diagnostic when it cannot recover.
- Solver calls publish convergence status into immutable training diagnostics;
  non-convergence is observable rather than being represented only by stderr.
- Randomness is passed through an explicit seeded generator.
- Expensive operations provide an optional preflight resource budget and fail
  before allocation when the conservative estimate exceeds that budget.
- Long-running training exposes cooperative cancellation at step boundaries;
  cancellation raises a typed domain exception before publishing a partial
  estimator state.
- Callers may also provide a positive wall-clock timeout; timeout expiry uses
  the same safe-boundary semantics as explicit cancellation.

## Documentation and style

- Public classes and functions use Google-style docstrings with `Args`,
  `Returns`, `Raises`, and `Examples` sections where applicable.
- Public examples are executable smoke tests, not independently maintained
  marketing snippets.
- Optional evaluation tooling may ship beside the core console entry points,
  but heavy evaluation dependencies remain lazy and absent from the core
  install. A future package split is a release-planning decision, not a reason
  to weaken the minimal-install import/help contract.
- Python changes must pass Ruff check/format and strict mypy. New public API
  requires documentation and a compatibility note.

## Testing and delivery

- Every bug fix adds a regression test at the narrowest useful level.
- Numerical changes add invariant/property coverage and a representative
  integration case.
- Release gates include clean install, wheel import, public examples, pickle
  round-trip, full tests, coverage, type checking, and site build.
- Performance-sensitive changes record runtime/memory impact with reproducible
  fixtures.
