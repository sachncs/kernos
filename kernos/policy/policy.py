"""Refresh controller: five-condition decision logic."""

from __future__ import annotations

from dataclasses import replace

from kernos.core.plan import Plan
from kernos.core.state import Bundle


class Policy:
    """Five-condition refresh controller."""

    def __init__(self, plan: Plan, drift_value: float) -> None:
        """Initialize a refresh decision from plan thresholds and drift."""
        self._plan = plan
        self._drift = drift_value

    def decide(self, bundle: Bundle, gain: float) -> bool:
        """Return ``True`` iff all five conditions are met.

        Conditions:
          1. ``drift > drift_hi``
          2. ``(step - tlast) >= cool``
          3. ``step >= warm``
          4. ``active == 1`` (or ``nohyst`` flag forces it on)
          5. ``gain > gain * rcost``
        """
        if self._plan.noref:
            return False
        if self._drift <= self._plan.drift_hi:
            return False
        if bundle.step - bundle.discrete.tlast < (0 if self._plan.nocool else self._plan.refresh_cooldown):
            return False
        if bundle.step < self._plan.refresh_warmup:
            return False
        active = 1 if self._plan.nohyst else bundle.discrete.active
        if active != 1:
            return False
        return gain > self._plan.refresh_gain_threshold * self._plan.refresh_cost

    def transition(self, bundle: Bundle) -> Bundle:
        """Update ``tlast`` to current step on refresh."""
        return bundle.replace(discrete=replace(bundle.discrete, tlast=bundle.step))
