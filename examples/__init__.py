"""Top-level examples package for Kernos.

This package is organised into two sub-packages:

- ``examples.evaluation`` — a full real-world evaluation suite with
  multiple datasets, budget tiers, ablations, and reporting
  (markdown, CSV, plots, LaTeX).
- ``examples.benchmarks`` — small, runnable demos showcasing the
  public API.

Both sub-packages are also exposed as console scripts via
``pyproject.toml``:

- ``kernos-eval`` — run the evaluation suite
- ``kernos-analyze`` — post-process a results CSV into plots and a
  LaTeX table
"""

from __future__ import annotations
