"""Amortized refresh budget accountant."""

from __future__ import annotations

from kernos.core.error import BudgetExceeded
from kernos.core.types import check


class Budget:
    """Tracks cumulative refresh cost against a fixed total budget."""

    def __init__(self, total: float) -> None:
        check(total > 0, f"total must be positive, got {total}")
        self.total = float(total)
        self.spent = 0.0

    def can(self, cost: float) -> bool:
        """Whether the next refresh of the given ``cost`` fits in the budget."""
        return self.spent + cost <= self.total

    def spend(self, cost: float) -> None:
        """Record a refresh of the given ``cost``.

        Raises:
            BudgetExceeded: When the resulting ``spent`` would exceed ``total``.
        """
        check(cost >= 0, f"cost must be non-negative, got {cost}")
        if not self.can(cost):
            raise BudgetExceeded(f"refresh cost {cost} would exceed remaining budget {self.remaining}")
        self.spent += cost

    @property
    def remaining(self) -> float:
        """Remaining budget, floored at 0."""
        return max(0.0, self.total - self.spent)
