"""LaTeX table exporter for main results (no ablations)."""

from __future__ import annotations

from pathlib import Path


def export_latex_table(rows: list[dict[str, str]], output_dir: Path) -> None:
    """Write ``results_table.tex`` containing Kernos + baselines only.

    Iterates over datasets × tiers × {Kernos, Ridge, Nystrom, RFF}.
    Rows for ablations are intentionally omitted.
    """
    datasets = sorted({r["dataset"] for r in rows})
    tiers = sorted({r["tier"] for r in rows})
    models = ["Kernos", "Ridge", "Nystrom", "RFF"]

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llllll}",
        "\\toprule",
        "Dataset & Tier & Model & RMSE & Train (s) & Peak Mem (MB) \\\\",
        "\\midrule",
    ]
    for dataset in datasets:
        for tier in tiers:
            for model in models:
                row = next(
                    (
                        r
                        for r in rows
                        if r["dataset"] == dataset and r["tier"] == tier and r["model"] == model
                    ),
                    None,
                )
                if row:
                    lines.append(
                        f"{dataset} & {tier} & {model} & "
                        f"{row['rmse']} & {row['train_time_sec']} & "
                        f"{row['peak_mem_mb']} \\\\"
                    )
        lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Evaluation results across datasets and budget tiers.}")
    lines.append("\\end{table}")

    out_path = output_dir / "results_table.tex"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {out_path}")
