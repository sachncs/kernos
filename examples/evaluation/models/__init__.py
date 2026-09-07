"""Model factories, budget tiers, and ablation presets for evaluation."""

from __future__ import annotations

from examples.evaluation.models.ablations import ABLATION_PRESETS
from examples.evaluation.models.factories import (
    make_kernos,
    make_nystrom,
    make_rff,
    make_ridge,
)
from examples.evaluation.models.tiers import (
    LARGE,
    MEDIUM,
    SMALL,
    TIER_MAP,
    BudgetTier,
)

__all__ = [
    "ABLATION_PRESETS",
    "BudgetTier",
    "LARGE",
    "MEDIUM",
    "SMALL",
    "TIER_MAP",
    "make_kernos",
    "make_nystrom",
    "make_rff",
    "make_ridge",
]
