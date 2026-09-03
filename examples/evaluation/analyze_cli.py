"""``kernos-analyze`` console script entry point.

Post-process a ``results.csv`` produced by ``kernos-eval`` into
markdown plots and a LaTeX table.

Usage::

    kernos-analyze --results results/results.csv --output_dir results
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from examples.evaluation.reporting.latex import export_latex_table
from examples.evaluation.reporting.plots import plot_ablation_bars, plot_pareto_fronts


def _parse_csv(path: Path) -> list[dict[str, str]]:
    """Parse ``path`` as a CSV into a list of row dicts."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    """CLI entry point for results analysis."""
    parser = argparse.ArgumentParser(description="Analyze Kernos evaluation results.")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results CSV.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for plots and LaTeX table.",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _parse_csv(results_path)
    print(f"Loaded {len(rows)} result rows from {results_path}")

    plot_pareto_fronts(rows, output_dir)
    plot_ablation_bars(rows, output_dir)
    export_latex_table(rows, output_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
