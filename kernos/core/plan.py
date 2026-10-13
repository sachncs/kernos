"""Validated compatibility configuration for the training runtime.

The flat frozen dataclass preserves the sklearn-facing constructor, while
immutable grouped configuration views expose cohesive boundaries to internal
and extension code. ``Plan.replace`` returns a new validated instance.

The grouped views expose these configuration boundaries without breaking the
flat estimator constructor:

* Embedding: ``dim``, ``seed``, ``embedder``.
* Basis capacity: ``mbasis``, ``abasis``, ``lk``, ``ltau``, ``basis``.
* Solver: ``ridge``, ``solver``, ``stab_*`` (numerical stability thresholds).
* Training: ``mode``, ``steps``, ``eval_every``, ``batch``, ``lr``,
  ``wr``, ``worth``, ``wdiv``, ``fdeps``.
* Refresh policy: ``drift_hi``, ``drift``, ``cool``, ``warm``, ``gain``,
  ``budget``, ``rcost``, ``amix``.
* Ablation: ``noref``, ``nohyst``, ``nocool``, ``noresid``, ``noorth``,
  ``nodiv``, ``nofreeze``, ``log_every``.
* Validation: ``val_frac``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import isfinite
from numbers import Integral, Real
from typing import Any

from kernos.core.config import (
    AblationConfig,
    BasisConfig,
    EmbeddingConfig,
    RefreshConfig,
    SolverConfig,
    TrainingConfig,
    ValidationConfig,
)
from kernos.core.types import check


class Buffer(Enum):
    """Cache mode for normal-equation accumulation."""

    FULL = "full"
    STREAM = "stream"
    ADAPTIVE = "adaptive"


VALID_EMBEDDERS = ("linear", "identity")
VALID_BASIS = ("nystrom", "greedy")
VALID_SOLVERS = ("direct", "iterative", "woodbury")
VALID_DRIFT = ("frobenius", "spectral")


@dataclass(frozen=True)
class Plan:
    """Master configuration for ``Kernos``."""

    dim: int = 64
    mbasis: int = 512
    abasis: int = 128
    ridge: float = 1e-3
    mode: Buffer = Buffer.FULL
    embedder: str = "linear"
    basis: str = "nystrom"
    solver: str = "direct"
    drift: str = "frobenius"

    steps: int = 1000
    eval_every: int = 10
    batch: int = 32
    seed: int | None = None

    lr: float = 1e-4
    wr: float = 0.0
    worth: float = 0.0
    wdiv: float = 0.0
    fdeps: float = 1e-5

    drift_hi: float = 0.1
    cool: int = 50
    warm: int = 10
    gain: float = 0.01
    budget: float = float("inf")
    memory_limit_mb: float | None = None
    rcost: float = 1.0
    amix: float = 0.5
    ltau: float = 0.1
    lk: int = 5

    noref: bool = False
    nohyst: bool = False
    nocool: bool = False
    noresid: bool = False
    noorth: bool = False
    nodiv: bool = False
    nofreeze: bool = False
    log_every: int = 0

    val_frac: float = 0.15

    stab_tau: float = 1e-6
    stab_alpha: float = 1e-5
    stab_eps: float = 1e-8
    stab_lmin: float = 1e-6
    stab_eta: float = 1e-4
    stab_kappa: float = 1e12
    stab_jitter: float = 1e-10
    stab_jitter_max: float = 1.0
    stab_jitter_retry: int = 5

    def __post_init__(self) -> None:
        """Validate configuration invariants after dataclass construction."""
        for name in ("dim", "mbasis", "abasis", "steps", "eval_every", "batch", "cool", "warm", "lk"):
            value = getattr(self, name)
            check(isinstance(value, Integral) and not isinstance(value, bool), f"{name} must be an integer")
            check(value >= 1 if name not in {"cool", "warm"} else value >= 0, f"{name} must be non-negative")
        for name in (
            "ridge",
            "lr",
            "wr",
            "worth",
            "wdiv",
            "fdeps",
            "drift_hi",
            "gain",
            "rcost",
            "amix",
            "ltau",
            "val_frac",
            "stab_tau",
            "stab_alpha",
            "stab_eps",
            "stab_lmin",
            "stab_eta",
            "stab_kappa",
            "stab_jitter",
            "stab_jitter_max",
        ):
            value = getattr(self, name)
            check(isinstance(value, Real) and not isinstance(value, bool), f"{name} must be a real number")
            check(isfinite(float(value)), f"{name} must be finite")
        check(isinstance(self.mode, Buffer), f"mode must be a Buffer, got {self.mode!r}")
        check(self.seed is None or isinstance(self.seed, Integral), "seed must be an integer or None")
        check(
            self.abasis <= 0.25 * self.mbasis,
            f"abasis ({self.abasis}) must be <= 0.25 * mbasis ({0.25 * self.mbasis})",
        )
        check(self.ridge >= self.stab_lmin, f"ridge ({self.ridge}) must be >= stab_lmin ({self.stab_lmin})")
        for name in (
            "stab_tau",
            "stab_alpha",
            "stab_eps",
            "stab_lmin",
            "stab_eta",
            "stab_kappa",
            "stab_jitter_max",
        ):
            check(getattr(self, name) > 0.0, f"{name} must be positive")
        check(self.stab_jitter >= 0.0, f"stab_jitter must be non-negative, got {self.stab_jitter}")
        check(
            self.stab_jitter_max >= self.stab_jitter,
            f"stab_jitter_max ({self.stab_jitter_max}) must be >= stab_jitter ({self.stab_jitter})",
        )
        check(
            isinstance(self.stab_jitter_retry, Integral)
            and not isinstance(self.stab_jitter_retry, bool)
            and self.stab_jitter_retry >= 0,
            "stab_jitter_retry must be a non-negative integer",
        )
        check(self.drift_hi > 0, f"drift_hi must be positive, got {self.drift_hi}")
        check(self.cool >= 0, f"cool must be non-negative, got {self.cool}")
        check(self.warm >= 0, f"warm must be non-negative, got {self.warm}")
        check(self.gain >= 0, f"gain must be non-negative, got {self.gain}")
        check(self.budget > 0, f"budget must be positive, got {self.budget}")
        check(
            self.memory_limit_mb is None
            or (
                isinstance(self.memory_limit_mb, Real)
                and not isinstance(self.memory_limit_mb, bool)
                and isfinite(float(self.memory_limit_mb))
                and self.memory_limit_mb > 0
            ),
            f"memory_limit_mb must be a finite positive number or None, got {self.memory_limit_mb}",
        )
        check(self.ltau > 0, f"ltau must be positive, got {self.ltau}")
        check(self.lk <= self.abasis, f"lk ({self.lk}) must be <= abasis ({self.abasis})")
        check(self.lk >= 1, f"lk must be >= 1, got {self.lk}")
        check(0 < self.val_frac < 1, f"val_frac must be in (0, 1), got {self.val_frac}")
        check(
            self.embedder in VALID_EMBEDDERS,
            f"embedder must be one of {VALID_EMBEDDERS}, got {self.embedder!r}",
        )
        check(self.basis in VALID_BASIS, f"basis must be one of {VALID_BASIS}, got {self.basis!r}")
        check(self.solver in VALID_SOLVERS, f"solver must be one of {VALID_SOLVERS}, got {self.solver!r}")
        check(self.drift in VALID_DRIFT, f"drift must be one of {VALID_DRIFT}, got {self.drift!r}")

    def replace(self, **kwargs: Any) -> Plan:
        """Return a new ``Plan`` with the given fields overridden."""
        return replace(self, **kwargs)

    @property
    def representation_regularization(self) -> float:
        """Return the regularization weight for the learned representation."""
        return self.wr

    @property
    def orthogonality_regularization(self) -> float:
        """Return the global/local orthogonality penalty weight."""
        return self.worth

    @property
    def diversity_regularization(self) -> float:
        """Return the local-feature diversity penalty weight."""
        return self.wdiv

    @property
    def finite_difference_step(self) -> float:
        """Return the perturbation magnitude used by the outer optimizer."""
        return self.fdeps

    @property
    def refresh_cooldown(self) -> int:
        """Return the minimum number of steps between refreshes."""
        return self.cool

    @property
    def refresh_warmup(self) -> int:
        """Return the minimum number of steps before refreshes are eligible."""
        return self.warm

    @property
    def refresh_gain_threshold(self) -> float:
        """Return the minimum gain multiplier required for a refresh."""
        return self.gain

    @property
    def refresh_cost(self) -> float:
        """Return the budget cost charged for one accepted refresh."""
        return self.rcost

    @property
    def anchor_mix(self) -> float:
        """Return the residual/coverage mixture used for anchor selection."""
        return self.amix

    @property
    def local_bandwidth(self) -> float:
        """Return the local RBF bandwidth multiplier."""
        return self.ltau

    @property
    def local_neighbors(self) -> int:
        """Return the number of local nearest anchors used per sample."""
        return self.lk

    @property
    def embedding_config(self) -> EmbeddingConfig:
        """Return the cohesive embedding configuration view."""
        return EmbeddingConfig(self.dim, self.embedder, self.seed)

    @property
    def basis_config(self) -> BasisConfig:
        """Return the cohesive basis configuration view."""
        return BasisConfig(self.mbasis, self.abasis, self.basis, self.ltau, self.lk)

    @property
    def solver_config(self) -> SolverConfig:
        """Return the cohesive solver configuration view."""
        return SolverConfig(
            self.ridge,
            self.solver,
            self.stab_tau,
            self.stab_alpha,
            self.stab_eps,
            self.stab_lmin,
            self.stab_eta,
            self.stab_kappa,
            self.stab_jitter,
            self.stab_jitter_max,
            self.stab_jitter_retry,
        )

    @property
    def training_config(self) -> TrainingConfig:
        """Return the cohesive continuous-training configuration view."""
        return TrainingConfig(
            self.mode,
            self.steps,
            self.eval_every,
            self.batch,
            self.lr,
            self.wr,
            self.worth,
            self.wdiv,
            self.fdeps,
        )

    @property
    def refresh_config(self) -> RefreshConfig:
        """Return the cohesive refresh-policy configuration view."""
        return RefreshConfig(
            self.drift,
            self.drift_hi,
            self.cool,
            self.warm,
            self.gain,
            self.budget,
            self.memory_limit_mb,
            self.rcost,
            self.amix,
        )

    @property
    def ablation_config(self) -> AblationConfig:
        """Return the cohesive experiment-switch configuration view."""
        return AblationConfig(
            self.noref,
            self.nohyst,
            self.nocool,
            self.noresid,
            self.noorth,
            self.nodiv,
            self.nofreeze,
            self.log_every,
        )

    @property
    def validation_config(self) -> ValidationConfig:
        """Return the cohesive validation configuration view."""
        return ValidationConfig(self.val_frac)

    def estimate_memory_bytes(self, n_samples: int, n_features: int) -> int:
        """Estimate peak working-set bytes for a batch.

        The estimate is intentionally conservative. It includes input and
        target storage plus either the fused feature matrix (full/adaptive)
        or the streamed normal equations. Callers can use it as a preflight
        guard before allocating the expensive feature representation.
        """
        check(n_samples >= 0, "n_samples must be non-negative")
        check(n_features >= 0, "n_features must be non-negative")
        base = n_samples * n_features * 8 + n_samples * 8
        feature_width = self.mbasis + self.abasis
        if self.mode == Buffer.STREAM:
            working = feature_width * feature_width * 8
        else:
            working = n_samples * feature_width * 8
        return int(2 * (base + working))
