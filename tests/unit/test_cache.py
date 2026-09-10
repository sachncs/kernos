"""Unit tests for kernos.cache."""

from __future__ import annotations

import numpy as np

from kernos.cache import Adaptive, Full, Stream


class TestFull:
    def test_accumulate(self, full_cache: Full, rng: np.random.Generator) -> None:
        phi = rng.standard_normal((10, 4))
        y = rng.standard_normal(10)
        full_cache.accumulate(phi, y)
        assert full_cache.size == 10

    def test_equations(self, full_cache: Full, rng: np.random.Generator) -> None:
        phi = rng.standard_normal((10, 4))
        y = rng.standard_normal(10)
        full_cache.accumulate(phi, y)
        S, b = full_cache.equations()
        np.testing.assert_allclose(S, phi.T @ phi)
        np.testing.assert_allclose(b, phi.T @ y)


class TestStream:
    def test_equations(self, stream_cache: Stream, rng: np.random.Generator) -> None:
        phi = rng.standard_normal((10, 8))
        y = rng.standard_normal(10)
        stream_cache.accumulate(phi, y)
        S, b = stream_cache.equations()
        np.testing.assert_allclose(S, phi.T @ phi)
        np.testing.assert_allclose(b, phi.T @ y)

    def test_count(self, stream_cache: Stream, rng: np.random.Generator) -> None:
        stream_cache.accumulate(rng.standard_normal((5, 8)), rng.standard_normal(5))
        assert stream_cache.count == 5


class TestAdaptive:
    def test_parity_full(self, adaptive_cache: Adaptive, rng: np.random.Generator) -> None:
        # Below threshold: behaves like Full
        phi = rng.standard_normal((10, 8))
        y = rng.standard_normal(10)
        adaptive_cache.accumulate(phi, y)
        S, b = adaptive_cache.equations()
        np.testing.assert_allclose(S, phi.T @ phi)

    def test_switches_to_stream(self, rng: np.random.Generator) -> None:
        cache = Adaptive(mfeat=8, threshold=5)
        cache.accumulate(rng.standard_normal((10, 8)), rng.standard_normal(10))
        assert isinstance(cache.primary, Stream)
