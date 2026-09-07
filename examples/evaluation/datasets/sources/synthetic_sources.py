"""Synthetic dataset generators."""

from __future__ import annotations

import numpy as np


def load_synthetic_spatial(
    rng: np.random.Generator,
    n: int = 5000,
    noise: float = 0.1,
    snr: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic 2D spatial field: smooth global + local high-frequency.

    Ground-truth field::

        f(x) = sin(2*pi*x1) * cos(2*pi*x2)           # smooth global
             + 0.5 * sin(20*pi*x1) * sin(20*pi*x2)   # local high-freq

    Noise scale is derived from ``snr`` (signal-to-noise ratio).
    The ``noise`` argument is reserved for future use and currently
    ignored.
    """
    del noise  # reserved for future use
    X = rng.uniform(0.0, 1.0, size=(n, 2))
    signal = np.sin(2.0 * np.pi * X[:, 0]) * np.cos(2.0 * np.pi * X[:, 1]) + 0.5 * np.sin(
        20.0 * np.pi * X[:, 0]
    ) * np.sin(20.0 * np.pi * X[:, 1])
    sig_var = float(np.var(signal))
    noise_std = np.sqrt(sig_var / snr) if snr > 0 else 0.0
    y = signal + noise_std * rng.standard_normal(n)
    return X.astype(np.float64), y.astype(np.float64)
