"""Check that release metadata agrees with the canonical package version."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_PATTERN = re.compile(r'__version__\s*=\s*["\']([^"\']+)["\']')


def main() -> None:
    """Exit non-zero when package and site release metadata diverge."""
    version_source = (ROOT / "kernos/_version.py").read_text(encoding="utf-8")
    match = VERSION_PATTERN.search(version_source)
    if match is None:
        raise SystemExit("Could not find canonical package version")
    version = match.group(1)

    package_version = json.loads((ROOT / "site/package.json").read_text(encoding="utf-8"))["version"]
    if package_version != version:
        raise SystemExit(f"site/package.json has {package_version}, expected {version}")

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if not re.search(rf"^## \[{re.escape(version)}\](?:\s|$)", changelog, re.MULTILINE):
        raise SystemExit(f"CHANGELOG.md has no release heading for {version}")

    checked_files = (
        "site/src/components/Footer.astro",
        "site/src/components/Hero.astro",
        "site/src/components/Install.astro",
    )
    for relative in checked_files:
        content = (ROOT / relative).read_text(encoding="utf-8")
        if version not in content:
            raise SystemExit(f"{relative} does not contain canonical version {version}")
    print(f"release metadata is synchronized at {version}")


if __name__ == "__main__":
    main()
