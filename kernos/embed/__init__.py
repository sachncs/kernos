"""Embed module: abstract ``Embed`` plus concrete ``Linear``, ``Kernel``, ``Identity`` and ``Projector``."""

from kernos.embed.identity import Identity
from kernos.embed.kernel import Kernel
from kernos.embed.linear import Linear
from kernos.embed.projector import Projector, norm

__all__ = ["Identity", "Kernel", "Linear", "Projector", "norm"]
