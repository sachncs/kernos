"""Plotting helpers: Pareto-front and ablation-bar visualisations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ABLATION_NAMES: list[str] = [
    "K-NoRefresh",
    "K-NoHysteresis",
    "K-NoCooldown",
    "K-NoResidAnchors",
    "K-NoOrthog",
    "K-NoDivPenalty",
    "K-NoFreeze",
]


def plot_pareto_fronts(rows: list[dict[str, str]], output_dir: Path) -> None:
    """Generate Pareto plots: RMSE vs Train Time per (dataset, tier)."""
    datasets = sorted({r["dataset"] for r in rows})
    tiers = sorted({r["tier"] for r in rows})
    colors = plt.cm.tab10(np.linspace(0, 1, 12))

    for dataset in datasets:
        fig, axes = plt.subplots(1, len(tiers), figsize=(5 * len(tiers), 5))
        if len(tiers) == 1:
            axes = [axes]

        for ax, tier in zip(axes, tiers, strict=False):
            subset = [r for r in rows if r["dataset"] == dataset and r["tier"] == tier]
            models = sorted({r["model"] for r in subset})
            for idx, model in enumerate(models):
                model_rows = [r for r in subset if r["model"] == model]
                rmse_parts = model_rows[0]["rmse"].split("±")
                time_parts = model_rows[0]["train_time_sec"].split("±")
                rmse_m = float(rmse_parts[0].strip())
                time_m = float(time_parts[0].strip())
                ax.scatter(time_m, rmse_m, label=model, color=colors[idx % 12], s=80)
                ax.annotate(model, (time_m, rmse_m), fontsize=6, alpha=0.7)

            ax.set_xlabel("Train Time (s)")
            ax.set_ylabel("RMSE")
            ax.set_title(f"{dataset} — {tier}")
            ax.legend(fontsize=5, loc="best")

        fig.tight_layout()
        out_path = output_dir / f"pareto_{dataset}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


def plot_ablation_bars(rows: list[dict[str, str]], output_dir: Path) -> None:
    """Bar plot of relative RMSE change per ablation vs full Kernos."""
    datasets = sorted({r["dataset"] for r in rows})
    tiers = sorted({r["tier"] for r in rows})

    for dataset in datasets:
        for tier in tiers:
            subset = [r for r in rows if r["dataset"] == dataset and r["tier"] == tier]
            full = next((r for r in subset if r["model"] == "Kernos"), None)
            if full is None:
                continue
            full_rmse = float(full["rmse"].split("±")[0].strip())

            names: list[str] = []
            rel_rmses: list[float] = []
            for ab in ABLATION_NAMES:
                row = next((r for r in subset if r["model"] == ab), None)
                if row:
                    names.append(ab.replace("K-", ""))
                    r = float(row["rmse"].split("±")[0].strip())
                    rel_rmses.append((r - full_rmse) / full_rmse * 100)

            if not names:
                continue

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(
                range(len(names)),
                rel_rmses,
                color=["red" if v > 0 else "green" for v in rel_rmses],
            )
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_ylabel("Rel. RMSE change (%)")
            ax.set_title(f"Ablations: {dataset} — {tier}")
            fig.tight_layout()
            out_path = output_dir / f"ablations_{dataset}_{tier}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Saved {out_path}")
