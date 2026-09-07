"""Quickstart demo: fit → predict on synthetic data.

Run with::

    python -m examples.benchmarks.quickstart
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kernos import Kernos

RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    """Generate synthetic data, fit, predict, and print metrics."""
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((200, 4))
    y_train = X_train[:, 0] + 0.5 * X_train[:, 1] ** 2 + 0.1 * rng.standard_normal(200)
    X_test = rng.standard_normal((50, 4))
    y_test = X_test[:, 0] + 0.5 * X_test[:, 1] ** 2 + 0.1 * rng.standard_normal(50)

    model = Kernos(seed=42, mbasis=24, abasis=4, lk=4).fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    print(f"R^2 score on held-out test set: {r2:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "quickstart.txt"
    out.write_text(f"quickstart: r2={r2:.6f}\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
