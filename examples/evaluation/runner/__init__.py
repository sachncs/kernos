"""Experiment execution: splits, single-run helpers, suite driver, diagnostics."""

from __future__ import annotations

from examples.evaluation.runner.diagnostics import compute_condition_proxy, tune_lambda_reg
from examples.evaluation.runner.single import run_single_baseline, run_single_kernos
from examples.evaluation.runner.splits import (
    TEST_FRAC,
    TRAIN_FRAC,
    VAL_FRAC,
    preprocess_and_split,
)
from examples.evaluation.runner.suite import run_experiment_suite

__all__ = [
    "TEST_FRAC",
    "TRAIN_FRAC",
    "VAL_FRAC",
    "compute_condition_proxy",
    "preprocess_and_split",
    "run_experiment_suite",
    "run_single_baseline",
    "run_single_kernos",
    "tune_lambda_reg",
]
