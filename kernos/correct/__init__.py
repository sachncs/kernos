"""Correct module: Sampler, Rbf, and orth sub-module."""

from kernos.correct.orth import Ridge, Tikhonov
from kernos.correct.rbf import Rbf
from kernos.correct.sampler import Sampler

__all__ = ["Rbf", "Ridge", "Sampler", "Tikhonov"]
