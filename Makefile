PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: setup test eval run clean

setup:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest

eval:
	$(PYTHON) -m pytest evals

run:
	$(PYTHON) -m agent.cli run --task "$(TASK)"

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov
