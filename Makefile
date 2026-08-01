PYTHON ?= $(shell python3 --version >/dev/null 2>&1 && echo python3 || echo python)
PIP ?= $(PYTHON) -m pip
TIMELOG ?= $(PYTHON) scripts/timelog.py

.PHONY: setup test eval run mockllm timelog clean

setup:
	$(TIMELOG) run --note "make setup" -- $(PIP) install -e ".[dev]"

test:
	$(TIMELOG) run --note "make test" -- $(PYTHON) -m pytest

eval:
	$(TIMELOG) run --note "make eval" -- $(PYTHON) -m evals.runner

run:
	$(TIMELOG) run --note "make run: $(TASK)" -- $(PYTHON) -m agent.user run --task "$(TASK)"

mockllm:
	$(PYTHON) -m mockllm.server --scenario "$(or $(SCENARIO),S01)"

timelog:
	$(TIMELOG) record --duration-hours "$(HOURS)" --note "$(NOTE)"

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov
