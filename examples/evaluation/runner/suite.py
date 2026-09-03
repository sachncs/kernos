"""Top-level experiment-suite driver.

Runs Kernos + baselines + (optionally) ablations for a single
``(dataset, tier)`` pair across multiple seeds.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from examples.evaluation.models.ablations import ABLATION_PRESETS
from examples.evaluation.models.factories import make_kernos, make_nystrom, make_rff, make_ridge
from examples.evaluation.models.tiers import BudgetTier
from examples.evaluation.reporting.result_record import ExperimentResult
from examples.evaluation.runner.diagnostics import tune_lambda_reg
from examples.evaluation.runner.single import run_single_baseline, run_single_kernos
from examples.evaluation.runner.splits import preprocess_and_split


def run_experiment_suite(
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    tier: BudgetTier,
    seeds: list[int],
    run_ablations: bool = True,
) -> list[ExperimentResult]:
    """Run Kernos, baselines, and ablations on ``(X, y)`` for one tier.

    For each seed the data is re-split and ``λ`` is re-tuned via
    :func:`tune_lambda_reg`. Returns a list of :class:`ExperimentResult`
    entries (one per model).
    """
    results: list[ExperimentResult] = []

    exp = ExperimentResult(dataset=dataset_name, model="Kernos", tier=tier.name)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        X_tr, X_val, X_te, y_tr, y_val, y_te = preprocess_and_split(X, y, rng)
        best_lam = tune_lambda_reg(
            lambda lam, _seed=seed: make_kernos(tier, lam, _seed),
            X_tr,
            y_tr,
            X_val,
            y_val,
            lambdas=[1e-4, 1e-3, 1e-2, 1e-1],
        )
        model = make_kernos(tier, best_lam, seed)
        exp.runs.append(run_single_kernos(model, X_tr, y_tr, X_te, y_te))
    results.append(exp)

    baseline_factories: dict[str, Callable[[float, int], object]] = {
        "Ridge": lambda lam, s: make_ridge(lam),
        "Nystrom": lambda lam, s: make_nystrom(tier, lam, s),
        "RFF": lambda lam, s: make_rff(tier, lam, s),
    }
    for bname, factory in baseline_factories.items():
        exp = ExperimentResult(dataset=dataset_name, model=bname, tier=tier.name)
        for seed in seeds:
            rng = np.random.default_rng(seed)
            X_tr, X_val, X_te, y_tr, y_val, y_te = preprocess_and_split(X, y, rng)
            best_lam = tune_lambda_reg(
                lambda lam, _seed=seed, _factory=factory: _factory(lam, _seed),
                X_tr,
                y_tr,
                X_val,
                y_val,
                lambdas=[1e-4, 1e-3, 1e-2, 1e-1, 1.0],
            )
            model = factory(best_lam, seed)
            exp.runs.append(run_single_baseline(model, X_tr, y_tr, X_te, y_te))
        results.append(exp)

    if run_ablations:
        for abname, abcfg in ABLATION_PRESETS.items():
            exp = ExperimentResult(dataset=dataset_name, model=abname, tier=tier.name)
            for seed in seeds:
                rng = np.random.default_rng(seed)
                X_tr, X_val, X_te, y_tr, y_val, y_te = preprocess_and_split(X, y, rng)
                best_lam = tune_lambda_reg(
                    lambda lam, _seed=seed, _abcfg=abcfg: make_kernos(
                        tier, lam, _seed, ablation=_abcfg
                    ),
                    X_tr,
                    y_tr,
                    X_val,
                    y_val,
                    lambdas=[1e-4, 1e-3, 1e-2, 1e-1],
                )
                model = make_kernos(tier, best_lam, seed, ablation=abcfg)
                exp.runs.append(run_single_kernos(model, X_tr, y_tr, X_te, y_te))
            results.append(exp)

    return results
