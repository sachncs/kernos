"""Check that the documented compatibility boundary names exist in code."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/api-reference.md"
NAME_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")

REQUIRED_NAMES = {
    "Kernos",
    "Buffer",
    "Lifecycle",
    "SolverEvent",
    "SolverStage",
    "RecoveryOutcome",
    "SolverRecovery",
    "TrainingDiagnostics",
    "MemoryLimitExceeded",
    "ShapeError",
    "Basis",
    "FeatureMap",
    "LocalFeatureMap",
    "Solver",
    "ResidualSolver",
    "Orthogonalizer",
    "FeatureFusion",
    "AnchorSampler",
    "WhiteningMap",
    "RefreshPipeline",
    "OuterStep",
    "BudgetAccountant",
    "RefreshPolicy",
    "DriftMetric",
}


def main() -> None:
    """Fail when a supported public name is undocumented or no longer exported."""
    documentation = DOC.read_text(encoding="utf-8")
    documented = set(NAME_PATTERN.findall(documentation))

    import kernos
    from kernos import core

    exported = set(kernos.__all__) | set(core.__all__)
    missing_exports = sorted(REQUIRED_NAMES - exported)
    missing_docs = sorted(REQUIRED_NAMES - documented)
    if missing_exports or missing_docs:
        details = []
        if missing_exports:
            details.append(f"missing exports: {', '.join(missing_exports)}")
        if missing_docs:
            details.append(f"missing docs: {', '.join(missing_docs)}")
        raise SystemExit("; ".join(details))
    print(f"API reference covers {len(REQUIRED_NAMES)} supported names")


if __name__ == "__main__":
    main()
