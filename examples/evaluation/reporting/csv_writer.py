"""CSV writer for experiment results."""

from __future__ import annotations

import csv
from pathlib import Path

from examples.evaluation.reporting.result_record import ExperimentResult


def write_results_csv(results: list[ExperimentResult], path: Path) -> None:
    """Write ``results`` to ``path`` as CSV.

    Raises ``IndexError`` if ``results`` is empty (no header to infer
    field names from). Caller is expected to guard with a non-empty
    result list.
    """
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].to_dict().keys())
        writer.writeheader()
        for exp in results:
            writer.writerow(exp.to_dict())
