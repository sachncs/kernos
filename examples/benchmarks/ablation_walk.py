"""Walk through each Kernos ablation flag on synthetic data.

Run with::

    python -m examples.benchmarks.ablation_walk
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kernos import Kernos
from kernos.bench.metric import rmse

RESULTS_DIR = Path(__file__).parent / "results"


ABLATION_FLAGS: list[tuple[str, str]] = [
    ("baseline", ""),
    ("noref", "noref"),
    ("nohyst", "nohyst"),
    ("nocool", "nocool"),
    ("noresid", "noresid"),
    ("noorth", "noorth"),
    ("nodiv", "nodiv"),
    ("nofreeze", "nofreeze"),
]


def main() -> None:
    """Fit Kernos with each ablation flag and report RMSE delta vs baseline."""
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((600, 4))
    y_train = X_train[:, 0] + 0.5 * X_train[:, 1] ** 2 + 0.1 * rng.standard_normal(600)
    X_test = rng.standard_normal((200, 4))
    y_test = X_test[:, 0] + 0.5 * X_test[:, 1] ** 2 + 0.1 * rng.standard_normal(200)

    rows: list[tuple[str, float, float]] = []
    for label, flag in ABLATION_FLAGS:
        kwargs: dict = dict(seed=42, mbasis=64, abasis=16, steps=50)
        if flag:
            kwargs[flag] = True
        model = Kernos(**kwargs).fit(X_train, y_train)
        r = rmse(y_test, model.predict(X_test))
        print(f"  {label:10s} RMSE={r:.4f}")
        rows.append((label, r, 0.0))

    base_rmse = rows[0][1]
    lines = ["| Ablation | RMSE | Δ vs baseline |", "|---|---|---|"]
    for label, r, _ in rows:
        delta = r - base_rmse
        lines.append(f"| {label} | {r:.4f} | {delta:+.4f} |")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "ablation_walk.md"
    out.write_text("# Ablation walk (synthetic, n=600, d=4)\n\n" + "\n".join(lines) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
