"""Result record dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SingleRunResult:
    """Metrics from a single (model, dataset, tier, seed) run."""

    rmse: float
    mae: float
    r2: float
    train_time_sec: float
    predict_time_sec: float
    peak_mem_mb: float
    refresh_count: int
    condition_proxy: float


@dataclass
class ExperimentResult:
    """Aggregated results for one (dataset, model, tier) across seeds."""

    dataset: str
    model: str
    tier: str
    runs: list[SingleRunResult] = field(default_factory=list)

    def mean_std(self, attr: str) -> tuple[float, float]:
        """Return ``(mean, std)`` of attribute ``attr`` across runs."""
        vals = [getattr(r, attr) for r in self.runs]
        return float(np.mean(vals)), float(np.std(vals))

    def to_dict(self) -> dict[str, object]:
        """Serialise to a flat dict suitable for CSV writing."""
        return {
            "dataset": self.dataset,
            "model": self.model,
            "tier": self.tier,
            "rmse": f"{self.mean_std('rmse')[0]:.6f} ± {self.mean_std('rmse')[1]:.6f}",
            "mae": f"{self.mean_std('mae')[0]:.6f} ± {self.mean_std('mae')[1]:.6f}",
            "r2": f"{self.mean_std('r2')[0]:.6f} ± {self.mean_std('r2')[1]:.6f}",
            "train_time_sec": (
                f"{self.mean_std('train_time_sec')[0]:.4f} ± {self.mean_std('train_time_sec')[1]:.4f}"
            ),
            "predict_time_sec": (
                f"{self.mean_std('predict_time_sec')[0]:.6f} ± {self.mean_std('predict_time_sec')[1]:.6f}"
            ),
            "peak_mem_mb": (f"{self.mean_std('peak_mem_mb')[0]:.2f} ± {self.mean_std('peak_mem_mb')[1]:.2f}"),
            "refresh_count": (
                f"{self.mean_std('refresh_count')[0]:.1f} ± {self.mean_std('refresh_count')[1]:.1f}"
            ),
            "condition_proxy": (
                f"{self.mean_std('condition_proxy')[0]:.2e} ± {self.mean_std('condition_proxy')[1]:.2e}"
            ),
            "stability_rmse_var": self.mean_std("rmse")[1],
            "stability_time_var": self.mean_std("train_time_sec")[1],
            "stability_cond_var": self.mean_std("condition_proxy")[1],
        }
