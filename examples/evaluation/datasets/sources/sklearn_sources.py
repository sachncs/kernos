"""scikit-learn built-in dataset loaders.

All loaders return ``(X, y)`` as ``np.ndarray`` of dtype ``float64``.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_diabetes as _sklearn_load_diabetes


def load_california_housing(*_args: object) -> tuple[np.ndarray, np.ndarray]:
    """California Housing: n≈20_640, d=8."""
    data = fetch_california_housing()
    return data.data.astype(np.float64), data.target.astype(np.float64)


def load_diabetes(*_args: object) -> tuple[np.ndarray, np.ndarray]:
    """Diabetes: n=442, d=10."""
    data = _sklearn_load_diabetes()
    return data.data.astype(np.float64), data.target.astype(np.float64)
