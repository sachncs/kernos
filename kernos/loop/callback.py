"""Callback ABC plus concrete ``Log``, ``Snapshot``, ``Profile``."""

from __future__ import annotations

import logging
import time

import numpy as np

from kernos.core.state import Bundle

logger = logging.getLogger("kernos")


class Callback:
    """Abstract callback interface."""

    def onstep(self, step: int, bundle: Bundle) -> None:
        """Called after every training step."""

    def oneval(self, step: int, metrics: dict[str, float]) -> None:
        """Called after every evaluation."""

    def onrefresh(self, step: int, bundle: Bundle) -> None:
        """Called after every refresh."""


class Log(Callback):
    """Logs metrics at ``log_every`` steps."""

    def __init__(self, log_every: int) -> None:
        self.log_every = log_every

    def onstep(self, step: int, bundle: Bundle) -> None:
        if self.log_every > 0 and step % self.log_every == 0:
            logger.info(
                "step=%d w_norm=%.3e",
                step,
                float(np.linalg.norm(bundle.weights)) if bundle.weights is not None else 0.0,
            )


class Snapshot(Callback):
    """Captures bundle snapshots at every refresh."""

    def __init__(self) -> None:
        self.snapshots: list[Bundle] = []

    def onrefresh(self, step: int, bundle: Bundle) -> None:
        self.snapshots.append(bundle)


class Profile(Callback):
    """Tracks wall-clock time per step and per refresh."""

    def __init__(self) -> None:
        self.step_times: list[float] = []
        self.refresh_times: list[float] = []
        self._tstep: float = 0.0
        self._trefresh: float = 0.0

    def onstep(self, step: int, bundle: Bundle) -> None:
        self._tstep = time.perf_counter()

    def oneval(self, step: int, metrics: dict[str, float]) -> None:
        self.step_times.append(time.perf_counter() - self._tstep)

    def onrefresh(self, step: int, bundle: Bundle) -> None:
        now = time.perf_counter()
        if self._trefresh > 0.0:
            self.refresh_times.append(now - self._trefresh)
        self._trefresh = now
