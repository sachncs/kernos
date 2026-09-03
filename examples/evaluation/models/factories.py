"""Model factories for Kernos and baselines."""

from __future__ import annotations

from kernos import Kernos
from kernos.bench.baseline import Nystrom as NystromBaseline
from kernos.bench.baseline import Random as RandomFeatureBaseline
from kernos.bench.baseline import Ridge as RidgeBaseline

from examples.evaluation.models.tiers import BudgetTier


def make_kernos(
    tier: BudgetTier,
    ridge: float,
    seed: int,
    ablation: dict[str, bool] | None = None,
) -> Kernos:
    """Construct a :class:`Kernos` estimator configured for a budget tier.

    Args:
        tier: Budget tier preset.
        ridge: Ridge regularization strength.
        seed: Random seed for reproducibility.
        ablation: Optional mapping of ``Plan`` flags (``noref``,
            ``nohyst``, ``nocool``, ``noresid``, ``noorth``,
            ``nodiv``, ``nofreeze``) to booleans. Unspecified flags
            default to ``False``.
    """
    ab = ablation or {}
    return Kernos(
        dim=tier.dim,
        mbasis=tier.mbasis,
        abasis=tier.abasis,
        ridge=ridge,
        mode=tier.mode,
        steps=tier.steps,
        eval_every=max(1, tier.steps // 20),
        seed=seed,
        budget=tier.total_refresh_budget,
        rcost=tier.refresh_cost,
        lr=1e-4,
        wr=1e-4,
        worth=1e-4,
        wdiv=1e-3,
        fdeps=1e-5,
        noref=ab.get("noref", False),
        nohyst=ab.get("nohyst", False),
        nocool=ab.get("nocool", False),
        noresid=ab.get("noresid", False),
        noorth=ab.get("noorth", False),
        nodiv=ab.get("nodiv", False),
        nofreeze=ab.get("nofreeze", False),
    )


def make_nystrom(tier: BudgetTier, ridge: float, seed: int) -> NystromBaseline:
    """Construct a Nyström baseline."""
    return NystromBaseline(mbasis=tier.mbasis, ridge=ridge, seed=seed)


def make_rff(tier: BudgetTier, ridge: float, seed: int) -> RandomFeatureBaseline:
    """Construct a Random Fourier Features baseline."""
    n_features = min(2000, max(tier.mbasis, 512))
    return RandomFeatureBaseline(mfeat=n_features, gamma=1.0, ridge=ridge, seed=seed)


def make_ridge(ridge: float) -> RidgeBaseline:
    """Construct a dense exact Ridge baseline."""
    return RidgeBaseline(ridge=ridge)
