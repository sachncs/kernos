"""Experiment runner and report dataclass."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kernos.bench.baseline import Baseline
from kernos.bench.metric import rmse


@dataclass
class Report:
    """A single experiment's metrics."""

    name: str
    rmse: float
    extra: dict[str, float]


class Runner:
    """Run one or many experiments and aggregate metrics."""

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed

    def run(self, name: str, model: Baseline, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Report:
        """Fit a single model and compute test RMSE."""
        model.fit(X_train, y_train)
        yhat = model.predict(X_test)
        return Report(name=name, rmse=rmse(y_test, yhat), extra={})

    def suite(self, models: dict[str, Baseline], X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> list[Report]:
        """Run a suite of models and return one ``Report`` per model."""
        return [self.run(name, model, X_train, y_train, X_test, y_test) for name, model in models.items()]

    def format(self, reports: list[Report]) -> str:
        """Format a markdown table of ``(name, rmse)`` rows."""
        rows = ["# Results", "", "| model | rmse |", "|---|---|"]
        for r in reports:
            rows.append(f"| {r.name} | {r.rmse:.4f} |")
        return "\n".join(rows)
