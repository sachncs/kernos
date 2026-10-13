"""Tests for registered strategy factories."""

from __future__ import annotations

import numpy as np
import pytest

from kernos.cache import Adaptive, Full, Stream
from kernos.core.factory import build_basis, build_cache
from kernos.core.plan import Buffer, Plan


def test_basis_factory_builds_registered_strategies(rng: np.random.Generator, whitening) -> None:
    U = rng.standard_normal((24, 4))
    nystrom = build_basis("nystrom", U, Plan(dim=4, mbasis=8, abasis=2, lk=2), whitening, rng)
    greedy = build_basis("greedy", U, Plan(dim=4, mbasis=8, abasis=2, lk=2), whitening, rng)

    assert nystrom.forward(U).shape[0] == U.shape[0]
    assert greedy.forward(U).shape[0] == U.shape[0]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [(Buffer.FULL, Full), (Buffer.STREAM, Stream), (Buffer.ADAPTIVE, Adaptive)],
)
def test_cache_factory_builds_registered_modes(mode: Buffer, expected: type) -> None:
    assert isinstance(build_cache(mode, mfeat=8, threshold=16), expected)
