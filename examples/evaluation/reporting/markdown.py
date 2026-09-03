"""Markdown table renderers for results and stability."""

from __future__ import annotations

from examples.evaluation.reporting.result_record import ExperimentResult


def results_to_markdown(results: list[ExperimentResult]) -> str:
    """Render a markdown table of mean±std metrics per (dataset, model, tier)."""
    lines = [
        "| Dataset | Model | Tier | RMSE | MAE | R^2 | Train(s) | PeakMem(MB) | Refresh | Cond(kappa) |",
        "|---------|-------|------|------|-----|-----|----------|-------------|---------|-------------|",
    ]
    for exp in results:
        d = exp.to_dict()
        lines.append(
            f"| {d['dataset']} | {d['model']} | {d['tier']} | {d['rmse']} | "
            f"{d['mae']} | {d['r2']} | {d['train_time_sec']} | "
            f"{d['peak_mem_mb']} | {d['refresh_count']} | {d['condition_proxy']} |"
        )
    return "\n".join(lines)


def stability_to_markdown(results: list[ExperimentResult]) -> str:
    """Render a markdown table of variance-over-refits per (dataset, model, tier)."""
    lines = [
        "| Dataset | Model | Tier | Var(RMSE) | Var(Time) | Var(Cond) |",
        "|---------|-------|------|-----------|-----------|-----------|",
    ]
    for exp in results:
        d = exp.to_dict()
        lines.append(
            f"| {d['dataset']} | {d['model']} | {d['tier']} | "
            f"{d['stability_rmse_var']:.6e} | {d['stability_time_var']:.6e} | "
            f"{d['stability_cond_var']:.6e} |"
        )
    return "\n".join(lines)
