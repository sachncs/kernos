"""Aggregate resource benchmark JSON files into a comparable summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

METRICS = (
    "fit_seconds",
    "refresh_fit_seconds",
    "predict_seconds",
    "rss_delta_mb",
    "estimated_memory_bytes",
)


def _parser() -> argparse.ArgumentParser:
    """Build the aggregation command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Benchmark JSON paths, or '-' for stdin.")
    return parser


def _read(path: str) -> dict[str, Any]:
    """Read one benchmark record from a file or stdin."""
    if path == "-":
        return json.load(sys.stdin)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    """Print host metadata and min/mean/max values for benchmark metrics."""
    records = [_read(path) for path in _parser().parse_args().paths]
    summary: dict[str, Any] = {
        "runs": len(records),
        "hosts": sorted({str(record["platform"]) for record in records}),
        "metrics": {},
    }
    for metric in METRICS:
        values = [float(record[metric]) for record in records]
        summary["metrics"][metric] = {
            "min": min(values),
            "mean": mean(values),
            "max": max(values),
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
