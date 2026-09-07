"""Budget tier presets aligned with the evaluation protocol."""

from __future__ import annotations

from dataclasses import dataclass

from kernos.core.plan import Buffer


@dataclass
class BudgetTier:
    """Preset budget tier for evaluation.

    Attributes:
        name: Display name (``"Small"``, ``"Medium"``, ``"Large"``).
        mbasis: Global basis rank budget (landmarks).
        abasis: Local corrective rank budget (anchors).
        steps: Maximum training steps.
        dim: Embedding dimension.
        total_refresh_budget: Amortized refresh budget.
        refresh_cost: Per-refresh cost charged against the budget.
        mode: :class:`Buffer` memory mode (default cached).
    """

    name: str
    mbasis: int
    abasis: int
    steps: int
    dim: int
    total_refresh_budget: float
    refresh_cost: float
    mode: Buffer = Buffer.FULL


SMALL = BudgetTier("Small", 128, 32, 200, 16, 5.0, 1.0)
MEDIUM = BudgetTier("Medium", 512, 128, 1000, 64, 20.0, 1.0)
LARGE = BudgetTier("Large", 2048, 512, 2000, 128, 50.0, 1.0)

TIER_MAP: dict[str, BudgetTier] = {"Small": SMALL, "Medium": MEDIUM, "Large": LARGE}
