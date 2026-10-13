"""Real-world evaluation suite for Kernos.

Sub-modules:

- :mod:`examples.evaluation.datasets` — dataset loaders and registry.
- :mod:`examples.evaluation.models` — budget tiers, model factories,
  ablation presets.
- :mod:`examples.evaluation.runner` — splits, single-run helpers,
  condition-proxy diagnostics, full experiment suite.
- :mod:`examples.evaluation.reporting` — result record types,
  markdown / CSV / plot / LaTeX writers.
- :mod:`examples.evaluation.cli` — ``kernos-eval`` console script.
- :mod:`examples.evaluation.analyze_cli` — ``kernos-analyze`` console
  script.
"""

from __future__ import annotations

__all__ = ["DATASET_REGISTRY"]


def __getattr__(name: str) -> object:
    """Load optional evaluation exports only when they are requested."""
    if name == "DATASET_REGISTRY":
        from examples.evaluation.datasets.registry import DATASET_REGISTRY

        return DATASET_REGISTRY
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
