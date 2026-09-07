"""Train/val/test splitting and optional standard scaling."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def preprocess_and_split(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    standardize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train/val/test with optional standard scaling.

    The split is a single permutation followed by fractional slicing.
    Targets are scaled only when ``standardize=True``.
    """
    n = X.shape[0]
    perm = rng.permutation(n)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    if standardize:
        scaler_x = StandardScaler()
        X_train = scaler_x.fit_transform(X_train)
        X_val = scaler_x.transform(X_val)
        X_test = scaler_x.transform(X_test)
        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
        y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    return X_train, X_val, X_test, y_train, y_val, y_test
