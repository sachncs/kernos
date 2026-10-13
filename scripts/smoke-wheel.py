"""Exercise the public estimator API from a clean package installation."""

from __future__ import annotations

import pickle

import numpy as np

from kernos import Kernos


def main() -> None:
    """Fit, predict, pickle and resume a small public estimator."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((40, 3))
    y = X[:, 0] - 0.25 * X[:, 1] + rng.standard_normal(40) * 0.01
    model = Kernos(dim=4, mbasis=8, abasis=2, lk=2, steps=1, seed=7, val_frac=0.2).fit(X, y)
    prediction = model.predict(X[:3])
    if prediction.shape != (3,) or not np.isfinite(prediction).all():
        raise SystemExit("clean-install prediction smoke test failed")
    restored = pickle.loads(pickle.dumps(model))
    restored.partial_fit(X, y)
    if restored.bundle_.step != model.bundle_.step + 1:
        raise SystemExit("clean-install pickle resume smoke test failed")
    print("clean-install fit/predict/pickle smoke test passed")


if __name__ == "__main__":
    main()
