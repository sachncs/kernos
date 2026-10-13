# Deployment

This document covers how to build, test, and publish kernos.

## Local Development

```bash
# Clone and install
git clone https://github.com/sachncs/kernos.git
cd kernos
uv sync --locked

# Run tests
pytest tests/ -v -p no:asyncio

# Run linters
ruff check .
ruff format --check .
mypy --strict kernos/
```

## Building Distribution

```bash
# Install build tools
pip install build twine

# Build source distribution and wheel
python -m build

# Verify the build
twine check dist/*
```

This creates a versioned wheel and source archive under `dist/`.

## Publishing to PyPI

### Test PyPI (recommended first)

```bash
# Upload to Test PyPI
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ kernos
```

### Production PyPI

Production publishing is performed by the tag-controlled trusted-publishing
workflow described below. Do not upload production artifacts manually from a
developer workstation.

## Release Checklist

1. Update the canonical version in `kernos/_version.py` only.
   Run `python scripts/check-version.py` to verify package and site metadata.
2. Update `CHANGELOG.md` with release date
3. Create a git tag:
   ```bash
   git tag -a vX.Y.Z -m "Release X.Y.Z"
   git push origin vX.Y.Z
   ```
4. Optionally build and upload to Test PyPI
5. Verify the test installation
6. Configure the `pypi` environment and trusted publisher
7. Push the reviewed `vX.Y.Z` tag to publish production artifacts
8. Create a GitHub Release with release notes

## Versioning

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality in a backwards-compatible manner
- **PATCH**: Backwards-compatible bug fixes

The version is tracked canonically in `kernos/_version.py`; setuptools reads it
from there and `scripts/check-version.py` guards the site/package metadata.

## GitHub Pages

Pushes to `master` or `main` trigger `.github/workflows/pages.yml`. The workflow
installs the locked site dependencies, runs the syntax/accessibility/content
smoke tests, builds the static site, deploys it, and fetches the public URL to
reject stale claims or artifacts. Use `workflow_dispatch` to redeploy without
creating a code change.

## Continuous PyPI Publishing

Releases are tag-controlled and use PyPI trusted publishing through
`.github/workflows/release.yml`. Configure the repository's `pypi` environment
and PyPI trusted publisher before creating a `vX.Y.Z` tag. The workflow checks
that the tag matches the canonical package version, validates both artifacts
with `twine check`, and publishes only after those gates pass.

The workflow is intentionally not exercised by local `make verify`, because
publishing requires the configured PyPI environment and a release tag.
