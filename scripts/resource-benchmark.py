"""Run a deterministic local resource benchmark for the public estimator."""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import sys
import time
from importlib.metadata import version

import numpy as np

try:
    import resource
except ImportError:  # pragma: no cover - exercised on Windows CI
    resource = None  # type: ignore[assignment]

from kernos import Kernos


def _rss_mb() -> float:
    """Return the current process high-water RSS in MiB across host OSes."""
    if sys.platform == "win32":

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        succeeded = ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
        if succeeded:
            return float(counters.PeakWorkingSetSize) / (1024.0 * 1024.0)
        return 0.0
    if resource is None:
        return 0.0
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return value / divisor


def _parser() -> argparse.ArgumentParser:
    """Build the benchmark command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-fit-seconds", type=float, default=60.0)
    parser.add_argument("--max-refresh-fit-seconds", type=float, default=60.0)
    parser.add_argument("--max-predict-seconds", type=float, default=10.0)
    parser.add_argument("--max-rss-delta-mb", type=float, default=256.0)
    return parser


def main() -> None:
    """Run the fixed workload and fail when configured resource budgets regress."""
    args = _parser().parse_args()
    rng = np.random.default_rng(123)
    X = rng.standard_normal((128, 4))
    y = X[:, 0] - 0.25 * X[:, 1] + rng.standard_normal(128) * 0.05
    model = Kernos(dim=4, mbasis=16, abasis=4, lk=4, steps=1, batch=16, seed=123)

    before_rss = _rss_mb()
    fit_started = time.perf_counter()
    model.fit(X, y)
    fit_seconds = time.perf_counter() - fit_started

    predict_started = time.perf_counter()
    predictions = model.predict(X)
    predict_seconds = time.perf_counter() - predict_started
    rss_delta_mb = max(0.0, _rss_mb() - before_rss)
    refresh_model = Kernos(
        dim=4,
        mbasis=16,
        abasis=4,
        lk=4,
        steps=1,
        eval_every=1,
        batch=16,
        seed=123,
        drift_hi=1e-9,
        cool=0,
        warm=0,
        budget=1.0,
        rcost=1.0,
    )
    refresh_started = time.perf_counter()
    refresh_model.fit(X, y)
    refresh_fit_seconds = time.perf_counter() - refresh_started
    result = {
        "fit_seconds": fit_seconds,
        "refresh_fit_seconds": refresh_fit_seconds,
        "predict_seconds": predict_seconds,
        "rss_delta_mb": rss_delta_mb,
        "estimated_memory_bytes": model.diagnostics_.estimated_memory_bytes,
        "steps": model.diagnostics_.steps,
        "refresh_evaluations": refresh_model.diagnostics_.evaluations,
        "refreshes": refresh_model.diagnostics_.refreshes,
        "predictions_finite": bool(np.isfinite(predictions).all()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "scikit_learn": version("scikit-learn"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if fit_seconds > args.max_fit_seconds:
        raise SystemExit(f"fit exceeded budget: {fit_seconds:.3f}s > {args.max_fit_seconds:.3f}s")
    if predict_seconds > args.max_predict_seconds:
        raise SystemExit(f"predict exceeded budget: {predict_seconds:.3f}s > {args.max_predict_seconds:.3f}s")
    if refresh_fit_seconds > args.max_refresh_fit_seconds:
        raise SystemExit(
            f"refresh fit exceeded budget: {refresh_fit_seconds:.3f}s > {args.max_refresh_fit_seconds:.3f}s"
        )
    if refresh_model.diagnostics_.evaluations != 1 or refresh_model.diagnostics_.refreshes < 1:
        raise SystemExit("refresh benchmark did not execute an accepted refresh")
    if rss_delta_mb > args.max_rss_delta_mb:
        raise SystemExit(f"RSS exceeded budget: {rss_delta_mb:.3f} MiB > {args.max_rss_delta_mb:.3f} MiB")
    if not result["predictions_finite"]:
        raise SystemExit("prediction output is not finite")


if __name__ == "__main__":
    main()
