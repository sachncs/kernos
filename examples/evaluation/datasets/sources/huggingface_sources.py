"""HuggingFace ``inria-soda/tabular-benchmark`` regression loaders."""

from __future__ import annotations

import numpy as np
from datasets import load_dataset


def _load_hf_regression(config: str, target_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Generic loader for ``inria-soda/tabular-benchmark`` regression configs."""
    ds = load_dataset("inria-soda/tabular-benchmark", config, split="train")
    df = ds.to_pandas()

    y = df[target_col].to_numpy(dtype=np.float64)
    X = df.drop(columns=[target_col]).to_numpy(dtype=np.float64)

    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[valid], y[valid]


def load_hf_wine_quality(*_args: object) -> tuple[np.ndarray, np.ndarray]:
    """Wine Quality: n≈6497, d=11."""
    return _load_hf_regression("reg_num_wine_quality", "quality")


def load_hf_abalone(*_args: object) -> tuple[np.ndarray, np.ndarray]:
    """Abalone: n≈4177, d=7."""
    return _load_hf_regression("reg_num_abalone", "ClassNumberOfRings")


def load_hf_superconduct(*_args: object) -> tuple[np.ndarray, np.ndarray]:
    """Superconduct: n≈21263, d=80."""
    return _load_hf_regression("reg_num_superconduct", "critical_temp")


def load_hf_house_sales(*_args: object) -> tuple[np.ndarray, np.ndarray]:
    """House Sales: n≈21613, d=15."""
    return _load_hf_regression("reg_num_house_sales", "price")


def load_hf_elevators(*_args: object) -> tuple[np.ndarray, np.ndarray]:
    """Elevators: n≈16599, d=18."""
    return _load_hf_regression("reg_num_elevators", "Elev")


def load_hf_cpu_act(*_args: object) -> tuple[np.ndarray, np.ndarray]:
    """CPU Activity: n≈8192, d=21."""
    return _load_hf_regression("reg_num_cpu_act", "usr")


def load_hf_diamonds(*_args: object) -> tuple[np.ndarray, np.ndarray]:
    """Diamonds: n≈53940, d=9 (with categorical encoding)."""
    ds = load_dataset("inria-soda/tabular-benchmark", "reg_cat_diamonds", split="train")
    df = ds.to_pandas()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].astype("category").cat.codes.astype(np.float64)
    y = df["price"].to_numpy(dtype=np.float64)
    X = df.drop(columns=["price"]).to_numpy(dtype=np.float64)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[valid], y[valid]
