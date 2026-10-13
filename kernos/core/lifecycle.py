"""Explicit estimator lifecycle states."""

from __future__ import annotations

from enum import Enum


class Lifecycle(Enum):
    """Observable lifecycle of a ``Kernos`` instance."""

    UNFITTED = "unfitted"
    FITTING = "fitting"
    FITTED = "fitted"
    STREAMING = "streaming"
