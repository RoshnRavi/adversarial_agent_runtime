PYTHON ?= $(shell python3 --version >/dev/null 2>&1 && echo python3 || echo python)
PIP ?= $(PYTHON) -m pip

.PHONY: setup test eval live-scenarios run mockllm clean

setup:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest

eval:
	$(PYTHON) -m evals.runner

live-scenarios:
	$(PYTHON) -m evals.live_scenarios

run:
	$(PYTHON) -m agent.user run --task "$(TASK)"

mockllm:
	$(PYTHON) -m mockllm.server --scenario "$(or $(SCENARIO),S01)"

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov
