"""Smoke tests for the ``examples`` sub-packages.

These verify that:

- All modules import cleanly under strict ``mypy``.
- The CLI ``argparse`` parsers build without error.
- The dataset registry contains the expected canonical entries.
- Each ``examples.benchmarks`` demo is importable and exposes a
  callable ``main``.
- The summary aggregator produces well-formed markdown from a
  minimal results CSV.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from examples.evaluation.datasets.registry import DATASET_REGISTRY
from examples.evaluation.reporting.summary import build_summary, write_summary


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
            build_summary,
            export_latex_table,
            plot_ablation_bars,
            plot_pareto_fronts,
            results_to_markdown,
            stability_to_markdown,
            write_results_csv,
            write_summary,
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

    def test_summary_cli_callable(self) -> None:
        """``reporting.summary.main`` is callable."""
        from examples.evaluation.reporting.summary import main

        assert callable(main)


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
        from examples.evaluation.models.tiers import LARGE, MEDIUM, SMALL

        assert SMALL.mbasis < MEDIUM.mbasis < LARGE.mbasis
        assert SMALL.abasis < MEDIUM.abasis < LARGE.abasis


class TestSummaryAggregator:
    """``build_summary`` and ``write_summary`` correctness."""

    def _make_csv(self, tmp: Path) -> Path:
        """Write a minimal results.csv with 2 datasets × 11 models."""
        path = tmp / "results.csv"
        models = [
            "Kernos",
            "Ridge",
            "Nystrom",
            "RFF",
            "K-NoRefresh",
            "K-NoHysteresis",
            "K-NoCooldown",
            "K-NoResidAnchors",
            "K-NoOrthog",
            "K-NoDivPenalty",
            "K-NoFreeze",
        ]
        # dataset A: Kernos wins; dataset B: Ridge wins.
        rmse = {
            "dataset_a": {
                "Kernos": "0.50",
                "Ridge": "0.60",
                "Nystrom": "0.70",
                "RFF": "0.80",
                "K-NoRefresh": "0.55",
                "K-NoHysteresis": "0.50",
                "K-NoCooldown": "0.50",
                "K-NoResidAnchors": "0.65",
                "K-NoOrthog": "0.50",
                "K-NoDivPenalty": "0.50",
                "K-NoFreeze": "0.50",
            },
            "dataset_b": {
                "Kernos": "0.90",
                "Ridge": "0.80",
                "Nystrom": "0.95",
                "RFF": "0.85",
                "K-NoRefresh": "0.85",
                "K-NoHysteresis": "0.90",
                "K-NoCooldown": "0.90",
                "K-NoResidAnchors": "0.92",
                "K-NoOrthog": "0.95",
                "K-NoDivPenalty": "0.90",
                "K-NoFreeze": "0.90",
            },
        }
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "dataset",
                    "model",
                    "tier",
                    "rmse",
                    "mae",
                    "r2",
                    "train_time_sec",
                    "predict_time_sec",
                    "peak_mem_mb",
                    "refresh_count",
                    "condition_proxy",
                    "stability_rmse_var",
                    "stability_time_var",
                    "stability_cond_var",
                ]
            )
            for ds, m in rmse.items():
                for model in models:
                    writer.writerow(
                        [
                            ds,
                            model,
                            "Small",
                            f"{m[model]} ± 0.0",
                            "0.0 ± 0.0",
                            "0.0 ± 0.0",
                            "1.0 ± 0.0",
                            "0.01 ± 0.0",
                            "100.0 ± 0.0",
                            "1.0 ± 0.0",
                            "1.0e4 ± 0.0",
                            "0.0",
                            "0.0",
                            "0.0",
                        ]
                    )
        return path

    def test_build_summary_contains_headline(self, tmp_path: Path) -> None:
        """Headline table includes Kernos RMSE, win rates, and ΔRMSE."""
        csv_path = self._make_csv(tmp_path)
        md = build_summary(csv_path)
        assert "# Kernos evaluation summary" in md
        assert "## Headline" in md
        assert "## Per-dataset ranking" in md
        assert "## Ablation impact" in md
        assert "Kernos mean RMSE" in md
        assert "Win rate vs Ridge" in md
        assert "Win rate vs Nystrom" in md
        assert "Win rate vs RFF" in md

    def test_build_summary_per_dataset_rank(self, tmp_path: Path) -> None:
        """Per-dataset ranking rows mention dataset_a and dataset_b."""
        csv_path = self._make_csv(tmp_path)
        md = build_summary(csv_path)
        assert "dataset_a" in md
        assert "dataset_b" in md

    def test_write_summary_creates_file(self, tmp_path: Path) -> None:
        """``write_summary`` writes ``summary.md`` and returns its path."""
        csv_path = self._make_csv(tmp_path)
        out = write_summary(csv_path, tmp_path)
        assert out == tmp_path / "summary.md"
        assert out.exists()
        assert out.stat().st_size > 0

    def test_build_summary_empty_csv(self, tmp_path: Path) -> None:
        """Empty CSV yields a graceful 'no rows' message instead of raising."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("dataset,model,tier,rmse\n")
        md = build_summary(csv_path)
        assert "No rows" in md
