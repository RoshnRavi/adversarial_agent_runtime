# Adversarial Agent Runtime

Custom Python agent runtime for the Part A adversarial-agent assessment.

## Overview

This repo contains a hand-rolled runtime for a hostile, Messages-API-shaped local mock
model. The Part A implementation lives under `agent/`. It persists run state in SQLite,
writes structured JSONL traces, exposes the required tools, and includes evals for the
assessment scenarios.

The framework rebuild shell lives under `agent_fw/`. It is only a Part B stub right now;
use `agent/cli.py` for the working Part A runtime.

## Requirements

- Python 3.10 or newer.
- SQLite, available through Python's standard `sqlite3` module.
- A terminal.
- For live `agent run` calls, a local mock model server that accepts the configured
  Messages-API-shaped request. The default endpoint is `http://localhost:8000/chat`.

`make test` and `make eval` do not require a live model server.

## Setup

```bash
# Install the package and development dependencies.
make setup
```

## Quick Start

```bash
# Run the scripted eval suite. This is the fastest no-server smoke check.
make eval

# Run the pytest suite directly.
make test

# Start a new Part A runtime task through the Makefile helper.
make run TASK="write a report"

# Start a new Part A runtime task directly.
python3 -m agent.cli run --task "write a report"

# Resume an interrupted run.
python3 -m agent.cli resume <run_id>

# Replay a recorded run without contacting the model server.
python3 -m agent.cli replay <run_id>
```

## CLI Reference

### Make Targets

```bash
# Install the editable package with dev dependencies.
make setup

# Run all pytest tests configured in pyproject.toml.
make test

# Run the custom eval runner, print pass rate, and compare with the stored baseline.
make eval

# Run a new agent task. TASK is passed to `agent.cli run --task`.
make run TASK="write a report"

# Remove local test and coverage caches.
make clean
```

### Part A Runtime

Use this CLI for the implemented custom runtime.

```bash
# Show available runtime commands and global flags.
python3 -m agent.cli --help

# Start a new task using the default database and mock server URL.
python3 -m agent.cli run --task "write a report"

# Start a new task with an explicit SQLite database and mock server URL.
python3 -m agent.cli --db runs/dev-agent.db --server-url http://localhost:8000/chat run --task "write a report"

# Resume a persisted run after interruption.
python3 -m agent.cli resume <run_id>

# Replay a run from its JSONL trace without the model server running.
python3 -m agent.cli replay <run_id>
```

Runtime flags:

- `--db`: SQLite database path. Default: `runs/agent_events.db`.
- `--server-url`: mock model endpoint. Default: `http://localhost:8000/chat`.

### Evals and Tests

```bash
# Run the custom eval suite directly.
python3 -m evals.runner

# Run pytest directly.
python3 -m pytest
```

### Chaos Harness

```bash
# Repeatedly launch a command and interrupt it at random times.
python3 harness/chaos.py --attempts 100 -- python3 -m agent.cli run --task "send exactly one email"

# Use a tighter interruption window.
python3 harness/chaos.py --attempts 25 --min-delay 0.05 --max-delay 0.25 -- python3 -m agent.cli run --task "write a file"
```

The local chaos helper sends `SIGTERM` first and escalates to `kill` only if the child
does not exit within its timeout.

### Part B Framework Stub

These commands currently mirror the intended Part B interface only. They print stub
messages and do not run the Part A implementation.

```bash
# Start a framework-backed run stub.
python3 -m agent_fw.cli run --task "write a report"

# Resume a framework-backed run stub.
python3 -m agent_fw.cli resume <run_id>

# Replay a framework-backed run stub.
python3 -m agent_fw.cli replay <run_id>
```

## Runtime Artifacts

- `runs/agent_events.db`: default SQLite event log and runtime state.
- `runs/traces/`: structured JSONL traces used by replay.
- `workspace/`: the only directory task file tools may write to.
- `evals/baseline.json`: stored eval baseline used by `make eval`.

## What Works

- Finite-state loop with durable SQLite run state and append-only events.
- SQLite WAL mode plus idempotency records for exactly-once simulated `send_email`.
- Tool boundary for `read_file`, `write_file`, `run_python`, `http_get`, and `send_email`.
- Workspace path confinement and HTTP host allow-listing.
- Deterministic 8,000-token memory budget using `mockllm/tokenizer.py`.
- JSONL traces and offline replay through `python3 -m agent.cli replay <run_id>`.
- Retry, exponential backoff, jitter, `Retry-After`, and circuit breaker support for mock-server calls.
- `make eval` runs 12 named S-cases and prints pass rate plus baseline diff.

## Known Gaps

- Tool calls are executed sequentially, not truly in parallel. Hangs are bounded by tool timeouts.
- The runtime records tool errors, but it does not yet challenge a later model claim that the failed tool succeeded.
- Partial interrupted parallel-turn reconstruction is incomplete.
- The local `mockllm/scenarios/` files in this repo are placeholders; the runtime is written for Messages-API-shaped local responses.
- The included chaos helper is useful for local interruption checks, but it is lighter than the assessment's full `kill -9` durability test.
- `agent_fw` is a framework CLI stub for Part B and is not a working framework rebuild yet.

## Eval Status

Current baseline: 9 passing cases, 3 expected failures.

Expected failures:

- S10 true parallel tool isolation.
- S11 model claim checked against tool error.
- S12 partial interrupted parallel turn recovery.
