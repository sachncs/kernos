"""Outer-loop loss functions used by ``Outerstep``.

Module-level functions — no class wrapper per the minimal-abstraction
rule.
"""

from __future__ import annotations

import numpy as np

from kernos.core.types import Array


def ridge(y: Array, phi: Array, weights: Array) -> float:
    """``0.5 * ||y - phi @ weights||^2 / n``."""
    samples = y.shape[0]
    residual = y - phi @ weights
    return float(0.5 * (residual @ residual) / samples)


def orth(R: Array) -> float:
    """``0.5 * ||R^T R - I||_F^2``."""
    dimension = R.shape[0]
    return float(0.5 * np.sum((R.T @ R - np.eye(dimension)) ** 2))


def div(phig: Array, phil: Array) -> float:
    """Diversity penalty: cosine distance between global and local blocks.

    Shape-agnostic: pads the smaller block with zeros to match shapes
    before computing the Frobenius inner product.
    """
    target_samples = max(phig.shape[0], phil.shape[0])
    target_features = max(phig.shape[1], phil.shape[1])
    g_padded = np.zeros((target_samples, target_features), dtype=np.float64)
    l_padded = np.zeros((target_samples, target_features), dtype=np.float64)
    g_padded[: phig.shape[0], : phig.shape[1]] = phig
    l_padded[: phil.shape[0], : phil.shape[1]] = phil
    ng = float(np.linalg.norm(g_padded, "fro"))
    nl = float(np.linalg.norm(l_padded, "fro"))
    if ng == 0.0 or nl == 0.0:
        return 0.0
    inner = float(np.sum(g_padded * l_padded))
    return 1.0 - inner / (ng * nl)


def outer(
    y: Array,
    phi: Array,
    weights: Array,
    R: Array,
    phig: Array,
    phil: Array,
    wr: float = 0.0,
    worth: float = 0.0,
    wdiv: float = 0.0,
) -> float:
    """Aggregate outer objective ``ridge + (wr + worth) * ||R||_F^2 + wdiv * div``.

    ``wr`` is the Frobenius regularizer weight and ``worth`` is the
    orthogonality penalty weight; both reduce to ``||R||``-norm terms
    so they are combined before scaling.
    """
    r_norm = float(np.linalg.norm(R, "fro"))
    return ridge(y, phi, weights) + (wr + worth) * r_norm * r_norm + wdiv * div(phig, phil)
