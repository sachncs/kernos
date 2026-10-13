"""Bounded runtime and native-memory smoke coverage."""

from __future__ import annotations

import resource
import sys
import time

import numpy as np
import pytest

from kernos import Kernos


def _max_rss_mb() -> float:
    """Return process high-water RSS in MiB on Unix platforms."""
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return value / divisor


@pytest.mark.integration
def test_small_fit_stays_within_resource_budget() -> None:
    """Keep a representative fit from regressing into an unbounded workload."""
    rng = np.random.default_rng(123)
    X = rng.standard_normal((96, 4))
    y = X[:, 0] - 0.25 * X[:, 1] + rng.standard_normal(96) * 0.05
    model = Kernos(dim=4, mbasis=16, abasis=4, lk=4, steps=1, batch=16, seed=123)

    before_rss = _max_rss_mb()
    started = time.perf_counter()
    model.fit(X, y)
    elapsed = time.perf_counter() - started
    rss_delta = max(0.0, _max_rss_mb() - before_rss)

    assert elapsed < 60.0
    assert rss_delta < 256.0
    assert model.diagnostics_.estimated_memory_bytes is not None
    assert model.diagnostics_.estimated_memory_bytes < 64 * 1024 * 1024
