"""Dataset registry: name → loader."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from examples.evaluation.datasets.sources.huggingface_sources import (
    load_hf_abalone,
    load_hf_cpu_act,
    load_hf_diamonds,
    load_hf_elevators,
    load_hf_house_sales,
    load_hf_superconduct,
    load_hf_wine_quality,
)
from examples.evaluation.datasets.sources.openml_sources import (
    load_kin8nm,
    load_nyc_taxi,
    load_protein,
    load_yearpredictionmsd,
)
from examples.evaluation.datasets.sources.sklearn_sources import (
    load_california_housing,
    load_diabetes,
)
from examples.evaluation.datasets.sources.synthetic_sources import load_synthetic_spatial

DATASET_REGISTRY: dict[str, Callable[..., tuple[np.ndarray, np.ndarray]]] = {
    # Sklearn / OpenML
    "CaliforniaHousing": load_california_housing,
    "Diabetes": load_diabetes,
    "Protein": load_protein,
    "Kin8nm": load_kin8nm,
    "YearPredictionMSD": load_yearpredictionmsd,
    "NYCTaxi": load_nyc_taxi,
    # HuggingFace tabular-benchmark
    "WineQuality": load_hf_wine_quality,
    "Abalone": load_hf_abalone,
    "Superconduct": load_hf_superconduct,
    "HouseSales": load_hf_house_sales,
    "Elevators": load_hf_elevators,
    "CPUActivity": load_hf_cpu_act,
    "Diamonds": load_hf_diamonds,
    # Synthetic
    "SyntheticSpatial": load_synthetic_spatial,
}
"""Mapping from dataset name to a loader returning ``(X, y)``."""
