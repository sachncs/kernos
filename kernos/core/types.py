"""Array type alias used throughout kernos."""

from typing import Any

import numpy as np

Array = np.ndarray[Any, Any]
"""Single-word alias for ``np.ndarray[Any, Any]`` used everywhere in kernos."""


def check(condition: bool, message: str) -> None:
    """Raise ``ValueError`` with ``message`` when ``condition`` is False.

    Single-word validation helper used in place of inline ``assert`` so error
    messages are user-facing and tests can target them.
    """
    if not condition:
        raise ValueError(message)


def asfloat64(x: Array) -> Array:
    """Return ``x`` cast to ``float64`` if it isn't already."""
    if x.dtype == np.float64:
        return x
    return x.astype(np.float64, copy=False)
