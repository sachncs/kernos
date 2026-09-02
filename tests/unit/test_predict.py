"""Unit tests for kernos.predict."""

from __future__ import annotations

import numpy as np
import pytest

from kernos.core.error import ShapeError
from kernos.predict.predict import Predict


class TestPredict:
    def test_forward(self, predict: Predict) -> None:
        phi = np.eye(4)
        out = predict.forward(phi)
        assert out.shape == (4,)

    def test_variance_no_sinv(self) -> None:
        p = Predict(weights=np.ones(4))
        with pytest.raises(ShapeError):
            p.variance(np.eye(4))

    def test_variance_with_sinv(self) -> None:
        p = Predict(weights=np.ones(4), Sinv=np.eye(4))
        out = p.variance(np.eye(4))
        assert out.shape == (4,)
        assert (out >= 0).all()
