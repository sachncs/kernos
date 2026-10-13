"""Check that every Plan field is represented by one immutable config view."""

from __future__ import annotations

from dataclasses import fields

from kernos.core.config import (
    AblationConfig,
    BasisConfig,
    EmbeddingConfig,
    RefreshConfig,
    SolverConfig,
    TrainingConfig,
    ValidationConfig,
)
from kernos.core.plan import Plan


def main() -> None:
    """Fail when the compatibility Plan and grouped views drift apart."""
    plan_fields = {field.name for field in fields(Plan)}
    view_types = (
        EmbeddingConfig,
        BasisConfig,
        SolverConfig,
        TrainingConfig,
        RefreshConfig,
        AblationConfig,
        ValidationConfig,
    )
    view_fields = {field.name for view_type in view_types for field in fields(view_type)}
    missing = sorted(plan_fields - view_fields)
    extra = sorted(view_fields - plan_fields)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing from views: {', '.join(missing)}")
        if extra:
            details.append(f"not present on Plan: {', '.join(extra)}")
        raise SystemExit("; ".join(details))
    print(f"configuration views cover {len(plan_fields)} Plan fields")


if __name__ == "__main__":
    main()
