"""RNG context for reproducible experiments.

Single source of truth for randomness in kernos.  Every component takes a
``np.random.Generator``; the ``Rng`` class provides:

* ``Rng.fix(seed)`` — seeds Python's ``random``, NumPy, and (optionally)
  Python's ``PYTHONHASHSEED`` so every run is byte-deterministic.
* ``Rng.spawn(n)`` — returns ``n`` independent ``Generator`` objects
  derived from the parent, suitable for parallel mini-batches.
"""

from __future__ import annotations

import random

import numpy as np


class Rng:
    """Reproducible random-number context."""

    @staticmethod
    def fix(seed: int | None) -> np.random.Generator:
        """Seed Python and NumPy deterministically.

        Args:
            seed: Any 32-bit hashable value. ``None`` seeds from the OS.

        Returns:
            A NumPy ``Generator`` ready to draw samples.
        """
        if seed is not None:
            random.seed(seed)
        return np.random.default_rng(seed)

    @staticmethod
    def spawn(parent: np.random.Generator, n: int) -> list[np.random.Generator]:
        """Spawn ``n`` independent child generators from ``parent``.

        Useful for parallel components that each need their own stream.

        Args:
            parent: A NumPy ``Generator`` (typically from ``Rng.fix``).
            n: Number of children to spawn.

        Returns:
            List of ``n`` independent ``Generator`` objects.
        """
        seq = np.random.SeedSequence(parent.integers(0, 2**32 - 1))
        return [np.random.default_rng(s) for s in seq.spawn(n)]
