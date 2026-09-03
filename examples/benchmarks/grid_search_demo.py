"""Demonstrate ``GridSearchCV`` integration with Kernos.

Run with::

    python -m examples.benchmarks.grid_search_demo
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import GridSearchCV

from kernos import Kernos

RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    """Run a small grid search over Kernos hyperparameters and print best params."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((300, 4))
    y = X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.1 * rng.standard_normal(300)

    param_grid = {"mbasis": [16, 32], "abasis": [4, 8], "ridge": [1e-2, 1e-1]}
    search = GridSearchCV(Kernos(seed=42, steps=50), param_grid, cv=3, n_jobs=1)
    search.fit(X, y)

    print(f"  best_params_: {search.best_params_}")
    print(f"  best_score_:  {search.best_score_:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "grid_search.txt"
    out.write_text(
        f"grid_search: best_params={search.best_params_} "
        f"best_score={search.best_score_:.6f}\n"
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
