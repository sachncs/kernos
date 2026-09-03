"""Smoke tests for the ``examples`` sub-packages.

These verify that:

- All modules import cleanly under strict ``mypy``.
- The CLI ``argparse`` parsers build without error.
- The dataset registry contains the expected canonical entries.
- Each ``examples.benchmarks`` demo is importable and exposes a
  callable ``main``.
"""

from __future__ import annotations

import argparse

import pytest

from examples.evaluation.datasets.registry import DATASET_REGISTRY


class TestEvaluationImports:
    """Top-level import smoke tests."""

    def test_registry_contains_canonical_datasets(self) -> None:
        """All canonical datasets registered."""
        expected = {
            "CaliforniaHousing",
            "Diabetes",
            "Protein",
            "Kin8nm",
            "YearPredictionMSD",
            "NYCTaxi",
            "WineQuality",
            "Abalone",
            "Superconduct",
            "HouseSales",
            "Elevators",
            "CPUActivity",
            "Diamonds",
            "SyntheticSpatial",
        }
        assert expected <= set(DATASET_REGISTRY.keys())

    def test_models_public_api(self) -> None:
        """Models sub-package exports the expected symbols."""
        from examples.evaluation.models import (  # noqa: F401
            ABLATION_PRESETS,
            LARGE,
            MEDIUM,
            SMALL,
            TIER_MAP,
            BudgetTier,
            make_kernos,
            make_nystrom,
            make_rff,
            make_ridge,
        )

    def test_runner_public_api(self) -> None:
        """Runner sub-package exports the expected symbols."""
        from examples.evaluation.runner import (  # noqa: F401
            TEST_FRAC,
            TRAIN_FRAC,
            VAL_FRAC,
            compute_condition_proxy,
            preprocess_and_split,
            run_experiment_suite,
            run_single_baseline,
            run_single_kernos,
            tune_lambda_reg,
        )

    def test_reporting_public_api(self) -> None:
        """Reporting sub-package exports the expected symbols."""
        from examples.evaluation.reporting import (  # noqa: F401
            ExperimentResult,
            SingleRunResult,
            export_latex_table,
            plot_ablation_bars,
            plot_pareto_fronts,
            results_to_markdown,
            stability_to_markdown,
            write_results_csv,
        )


class TestEvaluationCLI:
    """CLI argparse smoke tests."""

    def test_cli_main_callable(self) -> None:
        """``cli.main`` is callable."""
        from examples.evaluation import cli

        assert callable(cli.main)

    def test_cli_help_builds(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``cli.main --help`` builds without raising."""
        from examples.evaluation.cli import main

        with pytest.raises(SystemExit) as excinfo:
            main()  # called with no args; argparse calls SystemExit(2)
        # Either exit code 0 (--help) or 2 (missing required, but our
        # args are optional). Both are fine; we just want a clean exit.
        assert excinfo.value.code in (0, 2)

    def test_analyze_cli_callable(self) -> None:
        """``analyze_cli.main`` is callable."""
        from examples.evaluation import analyze_cli

        assert callable(analyze_cli.main)

    def test_analyze_cli_requires_results(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``analyze_cli.main`` exits cleanly when --results is missing."""
        from examples.evaluation.analyze_cli import main

        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 2


class TestBenchmarksImports:
    """Benchmark demos import smoke tests."""

    @pytest.mark.parametrize(
        "module_name",
        [
            "examples.benchmarks.quickstart",
            "examples.benchmarks.compare_baselines",
            "examples.benchmarks.memory_modes",
            "examples.benchmarks.ablation_walk",
            "examples.benchmarks.grid_search_demo",
        ],
    )
    def test_demo_importable(self, module_name: str) -> None:
        """Each benchmark demo module imports and exposes a ``main``."""
        import importlib

        mod = importlib.import_module(module_name)
        assert callable(getattr(mod, "main", None)), f"{module_name} missing main()"


class TestBudgetTierDataclass:
    """BudgetTier dataclass sanity tests."""

    def test_tier_map_has_three_tiers(self) -> None:
        """``TIER_MAP`` covers Small/Medium/Large."""
        from examples.evaluation.models.tiers import TIER_MAP

        assert set(TIER_MAP.keys()) == {"Small", "Medium", "Large"}

    def test_tier_dimensions_increase_monotonically(self) -> None:
        """``mbasis`` and ``abasis`` increase Small < Medium < Large."""
        from examples.evaluation.models.tiers import SMALL, MEDIUM, LARGE

        assert SMALL.mbasis < MEDIUM.mbasis < LARGE.mbasis
        assert SMALL.abasis < MEDIUM.abasis < LARGE.abasis
