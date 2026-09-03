"""Reporting: result records and markdown/CSV/plot/LaTeX writers."""

from __future__ import annotations

from examples.evaluation.reporting.csv_writer import write_results_csv
from examples.evaluation.reporting.latex import export_latex_table
from examples.evaluation.reporting.markdown import results_to_markdown, stability_to_markdown
from examples.evaluation.reporting.plots import plot_ablation_bars, plot_pareto_fronts
from examples.evaluation.reporting.result_record import ExperimentResult, SingleRunResult
from examples.evaluation.reporting.summary import build_summary, write_summary

__all__ = [
    "ExperimentResult",
    "SingleRunResult",
    "build_summary",
    "export_latex_table",
    "plot_ablation_bars",
    "plot_pareto_fronts",
    "results_to_markdown",
    "stability_to_markdown",
    "write_results_csv",
    "write_summary",
]
