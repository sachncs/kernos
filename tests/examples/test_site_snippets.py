"""Ensure Python snippets rendered by the site remain syntactically valid."""

import ast
import re
from pathlib import Path

SNIPPETS = Path(__file__).parents[2] / "examples" / "benchmarks" / "site_snippets.py"
MARKER = re.compile(r'# <snippet name="([^"]+)">\n([\s\S]*?)# </snippet>')


def test_site_snippets_compile() -> None:
    source = SNIPPETS.read_text(encoding="utf-8")
    blocks = dict(MARKER.findall(source))
    assert set(blocks) == {"fit", "grid", "stream"}
    for name, code in blocks.items():
        ast.parse(code.strip(), filename=f"site snippet: {name}")


def test_site_snippets_execute() -> None:
    source = SNIPPETS.read_text(encoding="utf-8")
    blocks = dict(MARKER.findall(source))
    for name, code in blocks.items():
        namespace = {"__name__": f"site_snippet_{name}"}
        exec(compile(code.strip(), f"site snippet: {name}", "exec"), namespace, namespace)
