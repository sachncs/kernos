"""Bench module: Baselines, datasets, metrics, runner."""

from kernos.bench.baseline import Baseline, Nystrom, Random, Ridge
from kernos.bench.dataset import hetero, highdim, linear, poly, split
from kernos.bench.metric import allmetrics, mae, maxerr, r2, rmse
from kernos.bench.runner import Report, Runner

__all__ = [
    "Baseline",
    "Nystrom",
    "Random",
    "Report",
    "Ridge",
    "Runner",
    "allmetrics",
    "hetero",
    "highdim",
    "linear",
    "mae",
    "maxerr",
    "poly",
    "r2",
    "rmse",
    "split",
]
