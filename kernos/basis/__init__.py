"""Basis module: Nyström, Random, Greedy and the Whitening helper."""

from kernos.basis.greedy import Greedy
from kernos.basis.nystrom import Nystrom
from kernos.basis.random import Random
from kernos.basis.whitening import Whitening

__all__ = ["Greedy", "Nystrom", "Random", "Whitening"]
