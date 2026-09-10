"""Unit tests for kernos.core.rng."""

from __future__ import annotations

import numpy as np

from kernos.core.rng import Rng


class TestFix:
    def test_returns_generator(self, rng: np.random.Generator) -> None:
        g = Rng.fix(42)
        assert isinstance(g, np.random.Generator)

    def test_deterministic(self) -> None:
        g1 = Rng.fix(7)
        g2 = Rng.fix(7)
        assert g1.integers(0, 100) == g2.integers(0, 100)


class TestSpawn:
    def test_spawn_count(self, rng: np.random.Generator) -> None:
        children = Rng.spawn(rng, 5)
        assert len(children) == 5
        for c in children:
            assert isinstance(c, np.random.Generator)

    def test_spawn_independence(self, rng: np.random.Generator) -> None:
        children = Rng.spawn(rng, 3)
        _ = children[0].integers(0, 1000)
        _ = children[1].integers(0, 1000)
        assert children[0] is not children[1]
