"""Check that documented README constructor defaults match ``Kernos``."""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from typing import Any

from kernos import Kernos

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
ROW = re.compile(r"^\| `(?P<name>[a-z_]+)` \| (?P<default>[^|]+) \|", re.MULTILINE)


def _parse_default(value: str) -> Any:
    """Parse a Markdown default cell into a comparable Python value."""
    normalized = value.strip().strip("`").replace("Buffer.FULL", "'full'")
    try:
        return ast.literal_eval(normalized)
    except (SyntaxError, ValueError) as exc:
        raise SystemExit(f"README default is not a literal: {value!r}") from exc


def main() -> None:
    """Fail when documented constructor defaults drift from the public signature."""
    signature = inspect.signature(Kernos)
    readme = README.read_text()
    configuration = readme.split("## Configuration", maxsplit=1)[1].split("### Ablation flags", maxsplit=1)[0]
    documented = {
        match.group("name"): _parse_default(match.group("default")) for match in ROW.finditer(configuration)
    }
    missing = sorted(name for name in documented if name not in signature.parameters)
    mismatches = []
    for name, actual in documented.items():
        if name in missing:
            continue
        expected = signature.parameters[name].default
        if expected != actual and not (expected is None and actual is None):
            mismatches.append(f"{name}: README={actual!r}, signature={expected!r}")
    if missing or mismatches:
        details = []
        if missing:
            details.append(f"unknown documented parameters: {', '.join(missing)}")
        if mismatches:
            details.append("default mismatches: " + "; ".join(mismatches))
        raise SystemExit("; ".join(details))
    print(f"README documents {len(documented)} synchronized constructor defaults")


if __name__ == "__main__":
    main()
