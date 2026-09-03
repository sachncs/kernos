"""Cross-dataset summary aggregator.

Reads ``results.csv`` produced by :mod:`examples.evaluation.cli` and
writes a ``summary.md`` containing:

- **Headline metrics** — Kernos vs each baseline, win rates, mean
  ΔRMSE vs the strongest baseline.
- **Per-dataset ranking** — for each dataset, where Kernos ranks out
  of all models (lower RMSE = better).
- **Ablation impact** — mean relative RMSE change for each ablation
  flag, averaged across datasets.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

BASELINES = ("Ridge", "Nystrom", "RFF")
ABLATIONS = (
    "K-NoRefresh",
    "K-NoHysteresis",
    "K-NoCooldown",
    "K-NoResidAnchors",
    "K-NoOrthog",
    "K-NoDivPenalty",
    "K-NoFreeze",
)


def _parse_mean(s: str) -> float:
    """Parse ``"0.823626 ± 0.000000"`` → ``0.823626``."""
    return float(s.split("±")[0].strip())


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    """Load a ``results.csv`` file as a list of row dicts."""
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _headline(rows: list[dict[str, str]]) -> list[str]:
    """Build the headline metrics markdown block."""
    kernos_rmses = np.array(
        [_parse_mean(r["rmse"]) for r in rows if r["model"] == "Kernos"]
    )
    if kernos_rmses.size == 0:
        return ["_No Kernos rows in CSV; cannot compute headline._", ""]
    kernos_mean = float(kernos_rmses.mean())

    best_base_name = "—"
    best_base_mean = float("inf")
    base_means: dict[str, float] = {}
    win_rates: dict[str, int] = {}
    n_datasets = len({r["dataset"] for r in rows if r["model"] == "Kernos"})

    for name in BASELINES:
        base_rmses = np.array([_parse_mean(r["rmse"]) for r in rows if r["model"] == name])
        if base_rmses.size == 0:
            continue
        m = float(base_rmses.mean())
        base_means[name] = m
        if m < best_base_mean:
            best_base_mean = m
            best_base_name = name
        wins = int(
            sum(
                1
                for ds in {r["dataset"] for r in rows if r["model"] == "Kernos"}
                if min(
                    _parse_mean(r["rmse"])
                    for r in rows
                    if r["dataset"] == ds and r["model"] == "Kernos"
                )
                < min(
                    _parse_mean(r["rmse"])
                    for r in rows
                    if r["dataset"] == ds and r["model"] == name
                )
            )
        )
        win_rates[name] = wins

    if best_base_mean == float("inf"):
        delta_str = "—"
    else:
        delta = (kernos_mean - best_base_mean) / best_base_mean * 100.0
        delta_str = f"{delta:+.2f}%"

    lines = [
        "| Metric | Value |",
        "|---|---|",
        f"| Kernos mean RMSE | {kernos_mean:.6f} |",
        f"| Best baseline mean RMSE | {best_base_mean:.6f} ({best_base_name}) |",
    ]
    for name in BASELINES:
        if name in win_rates:
            lines.append(f"| Win rate vs {name} | {win_rates[name]} / {n_datasets} |")
    lines.append(f"| Mean ΔRMSE vs best baseline | {delta_str} |")
    lines.append("")
    return lines


def _per_dataset_ranking(rows: list[dict[str, str]]) -> list[str]:
    """Build the per-dataset ranking markdown block."""
    datasets = sorted({r["dataset"] for r in rows})
    lines = [
        "| Dataset | Kernos rank | Top model | Worst model |",
        "|---|---|---|---|",
    ]
    for ds in datasets:
        per_model = [
            (r["model"], _parse_mean(r["rmse"]))
            for r in rows
            if r["dataset"] == ds
        ]
        per_model.sort(key=lambda kv: kv[1])
        if not per_model:
            continue
        models_in_order = [m for m, _ in per_model]
        kernos_rank = (
            models_in_order.index("Kernos") + 1 if "Kernos" in models_in_order else 0
        )
        lines.append(
            f"| {ds} | {kernos_rank} / {len(per_model)} | "
            f"{per_model[0][0]} | {per_model[-1][0]} |"
        )
    lines.append("")
    return lines


def _ablation_impact(rows: list[dict[str, str]]) -> list[str]:
    """Build the ablation-impact markdown block (relative ΔRMSE % vs Kernos)."""
    datasets = sorted({r["dataset"] for r in rows})
    lines = [
        "| Ablation | Mean ΔRMSE % |",
        "|---|---|",
    ]
    for ab in ABLATIONS:
        deltas: list[float] = []
        for ds in datasets:
            full = next(
                (r for r in rows if r["dataset"] == ds and r["model"] == "Kernos"),
                None,
            )
            ab_row = next(
                (r for r in rows if r["dataset"] == ds and r["model"] == ab),
                None,
            )
            if full is None or ab_row is None:
                continue
            full_rmse = _parse_mean(full["rmse"])
            ab_rmse = _parse_mean(ab_row["rmse"])
            if full_rmse > 0:
                deltas.append((ab_rmse - full_rmse) / full_rmse * 100.0)
        if deltas:
            lines.append(f"| {ab} | {float(np.mean(deltas)):+.2f}% |")
    lines.append("")
    return lines


def build_summary(csv_path: Path) -> str:
    """Build the full ``summary.md`` body as a string from ``csv_path``."""
    rows = _load_rows(csv_path)
    if not rows:
        return "# Kernos evaluation summary\n\n_No rows in results.csv._\n"

    datasets = sorted({r["dataset"] for r in rows})
    tiers = sorted({r["tier"] for r in rows})

    parts: list[str] = [
        "# Kernos evaluation summary",
        "",
        f"> Sweep over {len(datasets)} dataset(s) × tier(s) "
        f"{', '.join(tiers)}.",
        "",
        "## Headline",
        "",
    ]
    parts.extend(_headline(rows))
    parts.append("## Per-dataset ranking (1 = best)")
    parts.append("")
    parts.extend(_per_dataset_ranking(rows))
    parts.append("## Ablation impact (mean ΔRMSE % vs full Kernos)")
    parts.append("")
    parts.extend(_ablation_impact(rows))
    return "\n".join(parts)


def write_summary(csv_path: Path, output_dir: Path) -> Path:
    """Write ``summary.md`` to ``output_dir``; return the path written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "summary.md"
    out.write_text(build_summary(csv_path))
    return out


def main() -> None:
    """CLI entry point for ``python -m examples.evaluation.reporting.summary``."""
    parser = argparse.ArgumentParser(
        description="Compute and write summary.md from results.csv."
    )
    parser.add_argument("--results", type=str, required=True, help="Path to results.csv.")
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output directory."
    )
    args = parser.parse_args()

    csv_path = Path(args.results)
    output_dir = Path(args.output_dir)
    out = write_summary(csv_path, output_dir)
    print(out.read_text())


if __name__ == "__main__":
    main()
