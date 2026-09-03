"""OpenML / UCI tabular dataset loaders."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_openml


def load_protein(*_args: object) -> tuple[np.ndarray, np.ndarray]:
    """UCI Protein (OpenML 195): n=159, d=15."""
    data = fetch_openml(data_id=195, parser="auto", as_frame=False)
    return (
        data.data.astype(np.float64),
        data.target.astype(np.float64),
    )


def load_kin8nm(*_args: object) -> tuple[np.ndarray, np.ndarray]:
    """UCI kin8nm (OpenML 189): n=8192, d=8."""
    data = fetch_openml(data_id=189, parser="auto", as_frame=False)
    return (
        data.data.astype(np.float64),
        data.target.astype(np.float64),
    )


def load_yearpredictionmsd(*_args: object) -> tuple[np.ndarray, np.ndarray]:
    """YearPredictionMSD (OpenML 227): n≈8192, d=12.

    Note:
        This is the OpenML subsample; the full UCI version has ~515K
        samples.
    """
    data = fetch_openml(data_id=227, parser="auto", as_frame=False)
    return (
        data.data.astype(np.float64),
        data.target.astype(np.float64),
    )


def load_nyc_taxi(*_args: object) -> tuple[np.ndarray, np.ndarray]:
    """NYC Taxi Green Dec 2016 (OpenML 42729): n≈581_835, d=18.

    Large-scale benchmark for memory / runtime stress.
    """
    data = fetch_openml(data_id=42729, parser="auto", as_frame=False)
    return (
        data.data.astype(np.float64),
        data.target.astype(np.float64),
    )
