"""Single dataclass ``Plan`` holding every hyperparameter.

Per the design rule: minimal abstraction.  No sub-config classes.  All
fields live on one frozen dataclass; ``Plan.replace`` returns a new
instance with selected fields overridden.

Field groupings (documented only — they are not enforced structurally):

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

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

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
        check(self.abasis <= 0.25 * self.mbasis, f"abasis ({self.abasis}) must be <= 0.25 * mbasis ({0.25 * self.mbasis})")
        check(self.ridge >= self.stab_lmin, f"ridge ({self.ridge}) must be >= stab_lmin ({self.stab_lmin})")
        check(self.drift_hi > 0, f"drift_hi must be positive, got {self.drift_hi}")
        check(self.cool >= 0, f"cool must be non-negative, got {self.cool}")
        check(self.warm >= 0, f"warm must be non-negative, got {self.warm}")
        check(self.gain >= 0, f"gain must be non-negative, got {self.gain}")
        check(self.budget > 0, f"budget must be positive, got {self.budget}")
        check(self.ltau > 0, f"ltau must be positive, got {self.ltau}")
        check(self.lk <= self.abasis, f"lk ({self.lk}) must be <= abasis ({self.abasis})")
        check(self.lk >= 1, f"lk must be >= 1, got {self.lk}")
        check(0 < self.val_frac < 1, f"val_frac must be in (0, 1), got {self.val_frac}")
        check(self.embedder in VALID_EMBEDDERS, f"embedder must be one of {VALID_EMBEDDERS}, got {self.embedder!r}")
        check(self.basis in VALID_BASIS, f"basis must be one of {VALID_BASIS}, got {self.basis!r}")
        check(self.solver in VALID_SOLVERS, f"solver must be one of {VALID_SOLVERS}, got {self.solver!r}")
        check(self.drift in VALID_DRIFT, f"drift must be one of {VALID_DRIFT}, got {self.drift!r}")

    def replace(self, **kwargs: Any) -> "Plan":
        """Return a new ``Plan`` with the given fields overridden."""
        return replace(self, **kwargs)
