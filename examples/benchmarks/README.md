"""Benchmarks demos for Kernos.

Quick, runnable scripts showcasing the public API. Run any of them
with ``python -m examples.benchmarks.<name>``.

| Demo | What it shows |
|---|---|
| `quickstart` | Minimal fit → predict on synthetic data. |
| `compare_baselines` | Kernos vs Ridge vs Nystrom vs RFF on one dataset. |
| `memory_modes` | `Buffer.FULL` (cached) vs `Buffer.STREAM` comparison. |
| `ablation_walk` | One ablation flag at a time on synthetic data. |
| `grid_search_demo` | `GridSearchCV` integration with Kernos. |

All demos write a small result file under
`examples/benchmarks/results/` so multiple runs can be compared.
"""

from __future__ import annotations
