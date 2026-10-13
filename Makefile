PYTHON ?= .venv/bin/python
SITE_DIR := site

.PHONY: help test coverage coverage-check lint format-check typecheck api-check dependency-audit resource-benchmark package package-smoke site-build site-check site-visual verify

help:
	@printf '%s\n' \
		'test          Run the complete Python test suite' \
		'coverage      Run tests with the configured coverage report' \
		'coverage-check Run tests with the CI coverage threshold' \
		'lint          Run Ruff lint checks' \
		'format-check  Verify Ruff formatting' \
		'typecheck     Run strict mypy' \
		'api-check     Verify the documented public API boundary' \
		'dependency-audit Audit installed Python dependencies' \
		'resource-benchmark Run the deterministic local resource benchmark' \
		'package       Build wheel and sdist artifacts' \
		'package-smoke Build/install the wheel in a clean virtual environment' \
		'site-build    Build the static Astro site' \
		'site-check    Run site smoke, accessibility, browser and audit checks' \
		'site-visual   Compare reviewed desktop and phone screenshot baselines' \
		'verify        Run the local production-readiness gates'

test:
	$(PYTHON) -m pytest -q

coverage:
	$(PYTHON) -m pytest --cov=kernos --cov-report=term-missing

coverage-check:
	$(PYTHON) -m pytest tests/ --cov=kernos --cov-report=term-missing --cov-fail-under=80 -q

lint:
	$(PYTHON) -m ruff check kernos tests examples scripts

format-check:
	$(PYTHON) -m ruff format --check kernos tests examples scripts

typecheck:
	$(PYTHON) -m mypy kernos --strict

api-check:
	$(PYTHON) scripts/check-version.py
	$(PYTHON) scripts/check-api-reference.py
	$(PYTHON) scripts/check-config-views.py
	$(PYTHON) scripts/check-readme-config.py

dependency-audit:
	$(PYTHON) -m pip_audit

resource-benchmark:
	$(PYTHON) scripts/resource-benchmark.py

package:
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*

package-smoke: package
	@tmp_dir=$$(mktemp -d); \
	trap 'rm -rf "$$tmp_dir"' EXIT; \
	$(PYTHON) -m venv "$$tmp_dir/venv"; \
	"$$tmp_dir/venv/bin/python" -m pip install --quiet dist/kernos-*.whl; \
	"$$tmp_dir/venv/bin/python" scripts/smoke-wheel.py; \
	"$$tmp_dir/venv/bin/kernos-eval" --help >"$$tmp_dir/cli-help.out"; \
	grep -F 'Run the Kernos evaluation suite.' "$$tmp_dir/cli-help.out"; \
	set +e; \
	"$$tmp_dir/venv/bin/kernos-eval" --datasets Diabetes >"$$tmp_dir/cli.out" 2>&1; \
	exit_code=$$?; \
	set -e; \
	test "$$exit_code" -ne 0; \
	grep -F "pip install 'kernos[examples]'" "$$tmp_dir/cli.out"; \
	"$$tmp_dir/venv/bin/kernos-analyze" --help >"$$tmp_dir/analyze-help.out"; \
	grep -F 'Analyze Kernos evaluation results.' "$$tmp_dir/analyze-help.out"; \
	set +e; \
	"$$tmp_dir/venv/bin/kernos-analyze" --results "$$tmp_dir/results.csv" >"$$tmp_dir/analyze.out" 2>&1; \
	exit_code=$$?; \
	set -e; \
	test "$$exit_code" -ne 0; \
	grep -F "pip install 'kernos[examples]'" "$$tmp_dir/analyze.out"

site-build:
	npm --prefix $(SITE_DIR) run build

site-check:
	npm --prefix $(SITE_DIR) test
	npm --prefix $(SITE_DIR) run test:accessibility
	npm --prefix $(SITE_DIR) run test:browser
	npm --prefix $(SITE_DIR) audit --audit-level=high

site-visual:
	npm --prefix $(SITE_DIR) run test:visual

verify: lint format-check typecheck api-check dependency-audit coverage-check resource-benchmark package-smoke site-build site-check site-visual
