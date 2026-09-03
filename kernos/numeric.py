"""Numerical utilities for kernos.

Single-word module-level functions used throughout the codebase:
clipping, soft spectral truncation, dataset-scale epsilon, eigenvalue
retention, conditioning checks, Cholesky with jitter fallback, Frobenius
drift, and preconditioned conjugate gradient.
"""

from __future__ import annotations

import numpy as np

from kernos.core.error import IllConditionedError
from kernos.core.types import Array, check


def clipeig(evals: Array, floor: float = 0.0) -> Array:
    """Clip ``evals`` below at ``floor`` (default zero)."""
    return np.maximum(evals, floor)


def truncate(evals: Array, tau: float, eps: float) -> Array:
    """Soft spectral truncation.

    Eigenvalues below ``tau`` are down-weighted as
    ``max(tau, ev) / (max(tau, ev) + eps)`` to avoid discontinuities.
    """
    ev = np.maximum(evals, tau)
    return ev / (ev + eps)


def epsilon(tr: float, mbasis: int, alpha: float) -> float:
    """Dataset-scale epsilon for the whitening map.

    ``epsilon = alpha * tr / mbasis`` so the stabilization epsilon scales
    with the average eigenvalue of the kernel matrix.
    """
    check(mbasis > 0, f"mbasis must be > 0, got {mbasis}")
    return alpha * tr / mbasis


def retain(evals: Array, tau: float) -> tuple[Array, int]:
    """Indices of eigenvalues above ``tau`` and the count.

    Returns:
        (indices, rank) where ``rank = len(indices)``.
    """
    idx = np.where(evals > tau)[0]
    return idx, int(idx.size)


def condition(M: Array, kappa: float, name: str = "matrix") -> float:
    """Compute the condition number and raise if it exceeds ``kappa``.

    Args:
        M: Square matrix to inspect.
        kappa: Maximum acceptable condition number.
        name: Diagnostic name surfaced in the error message.

    Returns:
        The condition number.

    Raises:
        IllConditionedError: When ``cond(M) > kappa``.
    """
    condnum = float(np.linalg.cond(M))
    if condnum > kappa:
        raise IllConditionedError(f"{name} condition {condnum:.3e} exceeds kappa {kappa:.3e}")
    return condnum


def chol(M: Array, jitter: float = 1e-10, retry: int = 5, factor: float = 10.0, cap: float = 1.0) -> Array:
    """Cholesky factor with geometric jitter fallback.

    Args:
        M: Symmetric positive (semi-)definite matrix.
        jitter: Starting jitter added to the diagonal.
        retry: Maximum number of fallback attempts.
        factor: Multiplicative growth of jitter on each retry.
        cap: Maximum jitter value.

    Returns:
        Lower-triangular Cholesky factor.

    Raises:
        IllConditionedError: When all retries are exhausted.
    """
    eye = np.eye(M.shape[0], dtype=M.dtype)
    current = jitter
    for _ in range(retry + 1):
        try:
            return np.linalg.cholesky(M + current * eye)
        except np.linalg.LinAlgError:
            current = min(current * factor, cap)
    raise IllConditionedError(f"chol failed after {retry} retries (jitter={current})")


def fnorm(M: Array) -> float:
    """Frobenius norm of ``M``."""
    return float(np.linalg.norm(M, "fro"))


def drift(R: Array, Rref: Array) -> float:
    """Relative Frobenius-norm drift between ``R`` and ``Rref``.

    Returns 0.0 when ``Rref`` has zero Frobenius norm (avoids division by
    zero after the identity initialization).
    """
    delta = fnorm(R - Rref)
    baseline = fnorm(Rref)
    if baseline == 0.0:
        return 0.0
    return delta / baseline


def pcg(S: Array, b: Array, max_iter: int = 1000, tol: float = 1e-6, precon: Array | None = None) -> tuple[Array, int]:
    """Preconditioned CG solve of ``S @ x = b``.

    Args:
        S: Symmetric positive-definite matrix.
        b: Right-hand side.
        max_iter: Maximum iterations.
        tol: Convergence tolerance on the residual norm.
        precon: Preconditioner diagonal (left-preconditioner; identity if ``None``).

    Returns:
        (solution, info) where ``info > 0`` means non-convergence.
    """
    x = np.zeros_like(b)
    residual = b - S @ x
    z = precon * residual if precon is not None else residual
    direction = z.copy()
    rs_old = float(residual @ z)
    for _ in range(1, max_iter + 1):
        Sd = S @ direction
        alpha = rs_old / float(direction @ Sd)
        x = x + alpha * direction
        residual = residual - alpha * Sd
        if float(np.linalg.norm(residual)) < tol:
            return x, 0
        z = precon * residual if precon is not None else residual
        rs_new = float(residual @ z)
        beta = rs_new / rs_old
        direction = z + beta * direction
        rs_old = rs_new
    return x, max_iter
