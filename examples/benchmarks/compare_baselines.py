"""Compare Kernos against Ridge / Nystrom / RFF on one synthetic dataset.

Run with::

    python -m examples.benchmarks.compare_baselines
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kernos import Kernos
from kernos.bench.baseline import Nystrom, Random, Ridge
from kernos.bench.metric import rmse

RESULTS_DIR = Path(__file__).parent / "results"


def _make_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic regression problem and 70/30 split."""
    X = rng.standard_normal((1000, 5))
    y = X[:, 0] + 0.3 * X[:, 1] ** 2 - 0.5 * X[:, 2] + 0.1 * rng.standard_normal(1000)
    perm = rng.permutation(1000)
    train, test = perm[:700], perm[700:]
    return X[train], y[train], X[test], y[test]


def main() -> None:
    """Compare Kernos to three baselines and write a summary table."""
    rng = np.random.default_rng(7)
    X_train, y_train, X_test, y_test = _make_data(rng)

    models = {
        "Kernos": Kernos(seed=42, mbasis=64, abasis=16, steps=50),
        "Ridge": Ridge(ridge=1e-2),
        "Nystrom": Nystrom(mbasis=64, ridge=1e-2, seed=42),
        "RFF": Random(mfeat=512, gamma=1.0, ridge=1e-2, seed=42),
    }

    lines = ["| Model | RMSE |", "|---|---|"]
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r = rmse(y_test, y_pred)
        print(f"  {name:8s} RMSE={r:.4f}")
        lines.append(f"| {name} | {r:.4f} |")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "compare_baselines.md"
    out.write_text("# Baseline comparison (synthetic, n=1000, d=5)\n\n" + "\n".join(lines) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
