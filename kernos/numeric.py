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
    _validate_square_system(M)
    check(bool(np.isfinite(kappa)) and kappa > 0.0, "kappa must be finite and positive")
    condnum = float(np.linalg.cond(M))
    if condnum > kappa:
        raise IllConditionedError(f"{name} condition {condnum:.3e} exceeds kappa {kappa:.3e}")
    return condnum


def chol_with_jitter(
    M: Array, jitter: float = 1e-10, retry: int = 5, factor: float = 10.0, cap: float = 1.0
) -> tuple[Array, float, int]:
    """Return a Cholesky factor together with the jitter used and retry count.

    The metadata lets callers distinguish a clean factorization from a
    numerically stabilized fallback without parsing log messages.
    """
    _validate_square_system(M)
    check(bool(np.isfinite(jitter)) and jitter >= 0.0, "jitter must be finite and non-negative")
    check(isinstance(retry, int) and retry >= 0, "retry must be a non-negative integer")
    check(bool(np.isfinite(factor)) and factor > 1.0, "factor must be finite and greater than one")
    check(bool(np.isfinite(cap)) and cap >= jitter, "cap must be finite and at least jitter")
    eye = np.eye(M.shape[0], dtype=M.dtype)
    current = jitter
    for attempt in range(retry + 1):
        try:
            return np.linalg.cholesky(M + current * eye), current, attempt
        except np.linalg.LinAlgError:
            current = min(current * factor, cap)
    raise IllConditionedError(f"chol failed after {retry} retries (jitter={current})")


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
    factor_out, _, _ = chol_with_jitter(M, jitter=jitter, retry=retry, factor=factor, cap=cap)
    return factor_out


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


def pcg(
    S: Array, b: Array, max_iter: int = 1000, tol: float = 1e-6, precon: Array | None = None
) -> tuple[Array, int]:
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
    solution, info, _iterations = pcg_detailed(S, b, max_iter=max_iter, tol=tol, precon=precon)
    return solution, info


def pcg_detailed(
    S: Array, b: Array, max_iter: int = 1000, tol: float = 1e-6, precon: Array | None = None
) -> tuple[Array, int, int]:
    """Solve a symmetric system and return the actual iteration count.

    The legacy :func:`pcg` function keeps its two-value return contract;
    solver implementations that publish convergence telemetry can use this
    detailed variant.
    """
    _validate_square_shape(S)
    check(b.ndim == 1 and b.shape[0] == S.shape[0], "b must be a vector matching S")
    check(bool(np.isfinite(b).all()), "b must contain only finite values")
    check(isinstance(max_iter, int) and max_iter >= 1, "max_iter must be a positive integer")
    check(bool(np.isfinite(tol)) and tol > 0.0, "tol must be finite and positive")
    if not bool(np.isfinite(S).all()):
        return np.zeros_like(b), -1, 0
    check(np.allclose(S, S.T), "S must be symmetric")
    if precon is not None:
        check(precon.ndim == 1 and precon.shape[0] == S.shape[0], "precon must match S diagonal")
        check(
            bool(np.isfinite(precon).all()) and bool((precon > 0.0).all()),
            "precon must be finite and positive",
        )
    x = np.zeros_like(b)
    residual = b - S @ x
    z = precon * residual if precon is not None else residual
    direction = z.copy()
    rs_old = float(residual @ z)
    for iteration in range(1, max_iter + 1):
        Sd = S @ direction
        denominator = float(direction @ Sd)
        if not np.isfinite(denominator) or denominator <= 0.0:
            return x, -iteration, iteration
        alpha = rs_old / denominator
        x = x + alpha * direction
        residual = residual - alpha * Sd
        if not np.isfinite(x).all() or not np.isfinite(residual).all():
            return x, -iteration, iteration
        if float(np.linalg.norm(residual)) < tol:
            return x, 0, iteration
        z = precon * residual if precon is not None else residual
        rs_new = float(residual @ z)
        if not np.isfinite(rs_new) or rs_old == 0.0:
            return x, -iteration, iteration
        beta = rs_new / rs_old
        direction = z + beta * direction
        rs_old = rs_new
    return x, max_iter, max_iter


def _validate_square_system(matrix: Array) -> None:
    """Validate a finite non-empty square matrix before linear algebra."""
    _validate_square_shape(matrix)
    check(bool(np.isfinite(matrix).all()), "matrix must contain only finite values")


def _validate_square_shape(matrix: Array) -> None:
    """Validate only the shape needed to safely inspect a square matrix."""
    check(
        matrix.ndim == 2 and matrix.shape[0] > 0 and matrix.shape[0] == matrix.shape[1],
        "matrix must be non-empty and square",
    )
