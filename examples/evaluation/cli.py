"""``kernos-eval`` console script entry point.

Usage::

    kernos-eval --datasets WineQuality --tiers Small --n_seeds 2 --output_dir results

Or run the full sweep::

    kernos-eval --datasets all --tiers all --n_seeds 5 --output_dir results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from examples.evaluation.datasets.registry import DATASET_REGISTRY
from examples.evaluation.models.tiers import TIER_MAP
from examples.evaluation.reporting.csv_writer import write_results_csv
from examples.evaluation.reporting.markdown import results_to_markdown, stability_to_markdown
from examples.evaluation.reporting.result_record import ExperimentResult
from examples.evaluation.runner.suite import run_experiment_suite


def main() -> None:
    """CLI entry point for the evaluation suite."""
    parser = argparse.ArgumentParser(description="Run the Kernos evaluation suite.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Diabetes"],
        help="Dataset names or 'all'.",
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=["Small"],
        help="Budget tier names or 'all'.",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=5,
        help="Number of random seeds (default: 5).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory (default: results).",
    )
    parser.add_argument(
        "--no_ablations",
        action="store_true",
        help="Skip ablation runs.",
    )
    args = parser.parse_args()

    datasets = list(DATASET_REGISTRY.keys()) if args.datasets == ["all"] else args.datasets
    tiers = [TIER_MAP[t] for t in (["Small", "Medium", "Large"] if args.tiers == ["all"] else args.tiers)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[ExperimentResult] = []

    for dataset_name in datasets:
        if dataset_name not in DATASET_REGISTRY:
            print(f"WARNING: Unknown dataset '{dataset_name}', skipping.")
            continue
        print(f"\n=== Dataset: {dataset_name} ===")
        rng = np.random.default_rng(0)
        X, y = DATASET_REGISTRY[dataset_name](rng)
        print(f"  Loaded n={X.shape[0]}, d={X.shape[1]}")

        for tier in tiers:
            print(f"\n  -- Tier: {tier.name} --")
            seeds = [42 + i for i in range(args.n_seeds)]
            tier_results = run_experiment_suite(
                dataset_name, X, y, tier, seeds, run_ablations=not args.no_ablations
            )
            all_results.extend(tier_results)
            for exp in tier_results:
                rmse_m, rmse_s = exp.mean_std("rmse")
                print(
                    f"    {exp.model:22s} RMSE={rmse_m:.4f}±{rmse_s:.4f}  "
                    f"time={exp.mean_std('train_time_sec')[0]:.2f}s"
                )

    md_path = output_dir / "results.md"
    with open(md_path, "w") as f:
        f.write("# Evaluation Results\n\n")
        f.write("## Mean ± Std Metrics\n\n")
        f.write(results_to_markdown(all_results))
        f.write("\n\n## Stability-over-Refits\n\n")
        f.write(stability_to_markdown(all_results))
    print(f"\nWrote markdown: {md_path}")

    csv_path = output_dir / "results.csv"
    write_results_csv(all_results, csv_path)
    print(f"Wrote CSV: {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
