"""Refresh controller: five-condition decision logic."""

from __future__ import annotations

from kernos.core.plan import Plan
from kernos.core.state import Bundle


class Policy:
    """Five-condition refresh controller."""

    def __init__(self, plan: Plan, drift_value: float) -> None:
        self.plan = plan
        self.drift = drift_value

    def decide(self, bundle: Bundle, gain: float) -> bool:
        """Return ``True`` iff all five conditions are met.

        Conditions:
          1. ``drift > drift_hi``
          2. ``(step - tlast) >= cool``
          3. ``step >= warm``
          4. ``active == 1`` (or ``nohyst`` flag forces it on)
          5. ``gain > gain * rcost``
        """
        if self.plan.noref:
            return False
        if self.drift <= self.plan.drift_hi:
            return False
        if bundle.step - bundle.discrete.tlast < (0 if self.plan.nocool else self.plan.cool):
            return False
        if bundle.step < self.plan.warm:
            return False
        active = 1 if self.plan.nohyst else bundle.discrete.active
        if active != 1:
            return False
        return gain > self.plan.gain * self.plan.rcost

    def transition(self, bundle: Bundle) -> Bundle:
        """Update ``tlast`` to current step on refresh."""
        new_disc = bundle.discrete
        object.__setattr__(new_disc, "tlast", bundle.step)
        return bundle.replace(discrete=new_disc)
