"""Typed cohesive views over the public training configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kernos.core.plan import Buffer


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for the learned or identity feature map."""

    dim: int
    embedder: str
    seed: int | None


@dataclass(frozen=True)
class BasisConfig:
    """Configuration for landmark, anchor and local-feature capacity."""

    mbasis: int
    abasis: int
    basis: str
    ltau: float
    lk: int


@dataclass(frozen=True)
class SolverConfig:
    """Configuration for the normal-equation solver and stabilization."""

    ridge: float
    solver: str
    stab_tau: float
    stab_alpha: float
    stab_eps: float
    stab_lmin: float
    stab_eta: float
    stab_kappa: float
    stab_jitter: float
    stab_jitter_max: float
    stab_jitter_retry: int


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for continuous optimization and evaluation cadence."""

    mode: Buffer
    steps: int
    eval_every: int
    batch: int
    lr: float
    wr: float
    worth: float
    wdiv: float
    fdeps: float


@dataclass(frozen=True)
class RefreshConfig:
    """Configuration for drift detection, policy and refresh budgeting."""

    drift: str
    drift_hi: float
    cool: int
    warm: int
    gain: float
    budget: float
    memory_limit_mb: float | None
    rcost: float
    amix: float


@dataclass(frozen=True)
class AblationConfig:
    """Explicit feature switches for controlled experiments."""

    noref: bool
    nohyst: bool
    nocool: bool
    noresid: bool
    noorth: bool
    nodiv: bool
    nofreeze: bool
    log_every: int


@dataclass(frozen=True)
class ValidationConfig:
    """Configuration for validation partitioning."""

    val_frac: float
