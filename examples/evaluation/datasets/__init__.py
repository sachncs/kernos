"""Dataset loaders and registry for the evaluation suite.

All loaders return ``(X, y)`` as ``np.ndarray`` of dtype ``float64``.

Public surface:

- :data:`DATASET_REGISTRY` — mapping from dataset name (e.g.
  ``"WineQuality"``) to a zero-arg or ``rng``-arg callable returning
  ``(X, y)``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from examples.evaluation.datasets.registry import DATASET_REGISTRY

__all__ = ["DATASET_REGISTRY", "DatasetLoader"]

DatasetLoader = Callable[..., tuple[np.ndarray, np.ndarray]]
"""A dataset loader returns ``(X, y)`` after accepting optional ``rng``."""
