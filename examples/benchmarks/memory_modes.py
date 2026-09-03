"""Compare ``Buffer.FULL`` (cached) vs ``Buffer.STREAM`` (streamed) modes.

Run with::

    python -m examples.benchmarks.memory_modes
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kernos import Kernos
from kernos.bench.metric import rmse
from kernos.core.plan import Buffer

RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    """Fit Kernos in both memory modes on synthetic data and print timings."""
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((1000, 5))
    y_train = X_train[:, 0] + 0.3 * X_train[:, 1] ** 2 + 0.1 * rng.standard_normal(1000)
    X_test = rng.standard_normal((300, 5))
    y_test = X_test[:, 0] + 0.3 * X_test[:, 1] ** 2 + 0.1 * rng.standard_normal(300)

    lines = ["| Mode | RMSE |", "|---|---|"]
    for label, mode in (("FULL", Buffer.FULL), ("STREAM", Buffer.STREAM)):
        model = Kernos(seed=42, mode=mode, mbasis=64, abasis=16, steps=50)
        model.fit(X_train, y_train)
        r = rmse(y_test, model.predict(X_test))
        print(f"  {label:8s} RMSE={r:.4f}")
        lines.append(f"| {label} | {r:.4f} |")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "memory_modes.md"
    out.write_text("# Memory modes comparison (synthetic, n=1000, d=5)\n\n" + "\n".join(lines) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
