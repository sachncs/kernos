"""Factories for interchangeable runtime components.

The public estimator keeps compact string parameters for sklearn compatibility.
This module confines those names to one typed composition boundary so the
training loop depends on component contracts rather than scattered conditionals.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from kernos.basis.greedy import Greedy
from kernos.basis.nystrom import Nystrom
from kernos.basis.whitening import Whitening
from kernos.cache import Adaptive, Cache, Full, Stream
from kernos.core.interfaces import Basis, DriftMetric, FeatureMap, ResidualSolver
from kernos.core.plan import Buffer, Plan
from kernos.embed.identity import Identity
from kernos.embed.linear import Linear
from kernos.policy.drift import Frobenius, Spectral
from kernos.solver.direct import Direct
from kernos.solver.iterative import Iterative
from kernos.solver.woodbury import Woodbury

EmbedderFactory = Callable[[int, int, np.random.Generator], FeatureMap]
SolverFactory = Callable[[Plan], ResidualSolver]
DriftFactory = Callable[[], DriftMetric]
BasisFactory = Callable[[np.ndarray, Plan, Whitening, np.random.Generator], Basis]
CacheFactory = Callable[[int, int], Cache]


def _linear(input_dim: int, output_dim: int, rng: np.random.Generator) -> FeatureMap:
    return Linear(input_dim, output_dim, rng)


def _identity(input_dim: int, output_dim: int, rng: np.random.Generator) -> FeatureMap:
    del output_dim, rng
    return Identity()


def _direct(plan: Plan) -> ResidualSolver:
    return Direct(
        plan.ridge,
        plan.stab_jitter,
        plan.stab_jitter_retry,
        10.0,
        plan.stab_jitter_max,
        plan.stab_kappa,
    )


def _iterative(plan: Plan) -> ResidualSolver:
    return Iterative(plan.ridge)


def _woodbury(plan: Plan) -> ResidualSolver:
    return Woodbury(plan.ridge, plan.stab_jitter, plan.stab_jitter_retry, 10.0, plan.stab_jitter_max)


def _nystrom(U: np.ndarray, plan: Plan, whitening: Whitening, rng: np.random.Generator) -> Nystrom:
    return Nystrom.fromdata(U, plan.mbasis, whitening, rng)


def _greedy(U: np.ndarray, plan: Plan, whitening: Whitening, rng: np.random.Generator) -> Greedy:
    del whitening
    return Greedy.fromdata(U, plan.mbasis, rng)


def _full(mfeat: int, threshold: int) -> Cache:
    del mfeat, threshold
    return Full()


def _stream(mfeat: int, threshold: int) -> Cache:
    del threshold
    return Stream(mfeat=mfeat)


def _adaptive(mfeat: int, threshold: int) -> Cache:
    return Adaptive(mfeat=mfeat, threshold=threshold)


EMBEDDER_FACTORIES: dict[str, EmbedderFactory] = {"linear": _linear, "identity": _identity}
SOLVER_FACTORIES: dict[str, SolverFactory] = {
    "direct": _direct,
    "iterative": _iterative,
    "woodbury": _woodbury,
}
DRIFT_FACTORIES: dict[str, DriftFactory] = {"frobenius": Frobenius, "spectral": Spectral}
BASIS_FACTORIES: dict[str, BasisFactory] = {"nystrom": _nystrom, "greedy": _greedy}
CACHE_FACTORIES: dict[Buffer, CacheFactory] = {
    Buffer.FULL: _full,
    Buffer.STREAM: _stream,
    Buffer.ADAPTIVE: _adaptive,
}


def build_embedder(name: str, input_dim: int, output_dim: int, rng: np.random.Generator) -> FeatureMap:
    """Build an embedder from its registered name."""
    try:
        return EMBEDDER_FACTORIES[name](input_dim, output_dim, rng)
    except KeyError as exc:
        raise ValueError(f"Unknown embedder {name!r}; choose from {tuple(EMBEDDER_FACTORIES)}") from exc


def build_solver(plan: Plan) -> ResidualSolver:
    """Build a ridge solver from the plan's registered solver name."""
    try:
        return SOLVER_FACTORIES[plan.solver](plan)
    except KeyError as exc:
        raise ValueError(f"Unknown solver {plan.solver!r}; choose from {tuple(SOLVER_FACTORIES)}") from exc


def build_drift_metric(name: str) -> DriftMetric:
    """Build a drift metric from its registered name."""
    try:
        return DRIFT_FACTORIES[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown drift metric {name!r}; choose from {tuple(DRIFT_FACTORIES)}") from exc


def build_basis(
    name: str, U: np.ndarray, plan: Plan, whitening: Whitening, rng: np.random.Generator
) -> Basis:
    """Build a configured basis through the registered strategy factory."""
    try:
        return BASIS_FACTORIES[name](U, plan, whitening, rng)
    except KeyError as exc:
        raise ValueError(f"Unknown basis {name!r}; choose from {tuple(BASIS_FACTORIES)}") from exc


def restore_basis(
    name: str,
    landmarks: np.ndarray,
    whitening: np.ndarray | None,
    basis_wz: np.ndarray | None,
) -> Basis:
    """Restore a persisted basis snapshot using the registered strategy name."""
    if name == "greedy":
        if basis_wz is None:
            raise ValueError("Greedy basis requires basis_wz in the persisted snapshot")
        return Greedy(landmarks, basis_wz)
    if name == "nystrom":
        if whitening is None:
            raise ValueError("Nystrom basis requires whitening in the persisted snapshot")
        return Nystrom(landmarks, whitening)
    raise ValueError(f"Unknown basis {name!r}; choose from {tuple(BASIS_FACTORIES)}")


def build_cache(mode: Buffer, mfeat: int, threshold: int) -> Cache:
    """Build a normal-equation cache through the registered mode factory."""
    try:
        return CACHE_FACTORIES[mode](mfeat, threshold)
    except KeyError as exc:
        raise ValueError(f"Unknown cache mode {mode!r}; choose from {tuple(CACHE_FACTORIES)}") from exc
