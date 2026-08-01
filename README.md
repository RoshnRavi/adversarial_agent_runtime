# Adversarial Agent Runtime

Custom Python runtime for the Part A adversarial-agent assessment. The submitted
implementation lives in `agent/`; Part B is intentionally excluded until it is released.

## Current Status

- Preferred CLI: `python3 -m agent.user ...`
- Runtime defaults: `agent/config_agent.yaml`
- Persistence and replay: SQLite plus JSONL traces in `agent/run.py`
- Local tools: `read_file`, `write_file`, `run_python`, `http_get`, `send_email`
- Evals: 16 named cases, currently 14 passing and 2 expected failures
- Live `run` commands require a mock LLM server already listening at the configured
  `server_url`, default `http://localhost:8000/chat`

This checkout includes a small stdlib local server in `mockllm/server.py` for smoke tests
against the documented S1-S12 behaviours. If the official assessment mock server is
provided separately, use that server for final parity checks.

## Project Layout

- `agent/`: Part A runtime package.
  - `user.py`: user-facing CLI for `run`, `resume`, and `replay`.
  - `runtime.py`: run IDs, loop/step control, runtime config, DTOs.
  - `message.py`: message window, tool-result messages, context compaction.
  - `validate.py`: mock LLM HTTP client, retries, circuit breaker, per-turn validation.
  - `response.py`: parses final text, tool calls, and malformed response parts.
  - `executer.py`: local tool execution and safety checks.
  - `run.py`: SQLite state/events, pending tool calls, JSONL trace, replay.
  - `config_agent.yaml`: default paths, limits, server URL, and tool policy.
- `evals/`: pytest tests, eval runner, YAML scripted inputs, case definitions, and baseline.
- `harness/`: chaos helper for interruption testing.
- `mockllm/`: deterministic tokenizer, scenario YAML, and local HTTP mock server.
- `scripts/`: support scripts, currently `timelog.py`.
- `runs/`: runtime databases and JSONL traces.
- `workspace/`: confined filesystem sandbox for task file tools.

## Requirements

- Python 3.10 or newer.
- SQLite through Python's standard `sqlite3` module.
- PyYAML, installed from project dependencies.
- A terminal.
- A separate local mock LLM server for live `run` commands.

`make test`, `make eval`, and `replay` do not require the mock server.

## Setup

```bash
make setup
```

`make setup` installs the package in editable mode with development dependencies.

## Quick Start

No-server checks:

```bash
make test
make eval
agent replay <actual_run_id>
python3 -m agent.user replay <actual_run_id>
```

Live run, only after a mock server is listening at `server_url`:

```bash
python3 -m agent.user run --task "write a report"
```

Use the printed `run_id` for resume and replay:

```bash
python3 -m agent.user resume b4b4b7fd-9768-44c5-ba01-7410a5b90e35
agent replay b4b4b7fd-9768-44c5-ba01-7410a5b90e35
python3 -m agent.user replay b4b4b7fd-9768-44c5-ba01-7410a5b90e35
```

Do not include angle brackets literally; `<run_id>` in examples means replace it with
the actual ID printed by `run`.

## Make Targets

```bash
make setup
make test
make eval
make mockllm SCENARIO=S01
make run TASK="write a report"
make timelog HOURS="0.25" NOTE="design review and write-up"
make clean
```

`setup`, `test`, `eval`, and `run` are wrapped by `scripts/timelog.py`, so they update
`TIMELOG.md` automatically after the command finishes. Failed commands are logged too,
because they still count as assessment work.

`make mockllm SCENARIO=S01` starts the local mock server at
`http://127.0.0.1:8000/chat`. Use `S01` through `S12` for the documented scenario
shapes.

## Runtime CLI

```bash
python3 -m agent.user --help
python3 -m agent.user run --task "write a file"
python3 -m agent.user run --run-id r2-check --task "send exactly one email"
python3 -m agent.user resume <actual_run_id>
agent replay <actual_run_id>
python3 -m agent.user replay <actual_run_id>
```

Optional global overrides:

```bash
python3 -m agent.user \
  --db runs/dev-agent.db \
  --server-url http://localhost:8000/chat \
  run --task "write a report"
```

- `--db`: overrides the SQLite database path from `agent/config_agent.yaml`.
- `--server-url`: overrides the mock model endpoint from `agent/config_agent.yaml`.

If a live run fails with:

```text
NetworkFailureError: mock server failed: <urlopen error [Errno 111] Connection refused>
```

then no process is listening at the configured mock server URL. Start the assessment
mock server, run `make mockllm SCENARIO=S01`, or change `server_url` to the correct
endpoint.

## Configuration

Defaults live in `agent/config_agent.yaml`:

```yaml
db_path: runs/agent_events.db
trace_dir: runs/traces
workspace_root: workspace
server_url: http://localhost:8000/chat
token_budget: 8000
cost_budget_tokens: 50000
max_steps: 50
max_retries: 4
retry_base_delay: 0.25
request_timeout_seconds: 5.0
circuit_max_failures: 5
circuit_cooldown_seconds: 60
no_progress_limit: 3
python_timeout_seconds: 5
python_memory_mb: 64
http_timeout_seconds: 10.0
http_allow_hosts:
  - localhost
  - 127.0.0.1
```

The loader in `agent/runtime.py` validates and coerces the YAML values into
`RuntimeConfig`. Path values remain relative to the process working directory.
`token_budget` limits the compacted context window and is capped at the required
8,000-token ceiling measured by `mockllm/tokenizer.py`; smaller values are useful for
tests. `cost_budget_tokens` is a cumulative simulated run-cost ceiling charged from
deterministic request and response token counts.

## Evals And Tests

```bash
python3 -m pytest
python3 -m evals.runner
```

`make eval` runs the same eval runner and compares the result with `evals/baseline.json`.
Scripted eval inputs live in `evals/input.yaml`; case names, adversarial flags, and
expected-failure explanations live in `evals/cases.py`. The current baseline is:

- 16 total cases
- 14 passing cases
- 2 expected failures
- pass rate 0.875

Two passing evals (`I01` and `I02`) exercise the runtime through the real HTTP client
against the local `mockllm.server`; the rest use deterministic scripted responses for
speed and failure isolation.

Expected failures are real executed evals, not skipped cases:

- FAIL01: generic false success claim without a concrete failed tool target.
- FAIL02: transactional rollback across mixed tool batches.

## Observability And Replay

Every run persists state and events to SQLite through `agent/run.py`. Structured JSONL
traces are written under `runs/traces/` and can be replayed without the mock server:

```bash
agent replay <actual_run_id>
python3 -m agent.user replay <actual_run_id>
```

Replay reconstructs recorded model/tool/final decisions from the JSONL trace only. It
does not contact the model server, execute tools, or repeat side effects.

## Time Log

`TIMELOG.md` is updated automatically by:

- `make setup`
- `make test`
- `make eval`
- `make run`

Manual entries:

```bash
make timelog HOURS="0.25" NOTE="triage Part A gaps"
python3 scripts/timelog.py record --duration-hours 0.25 --note "write-up"
```

Command durations are rounded up to the nearest `0.05h` and merged into one row per
local date.

## Chaos Harness

```bash
python3 harness/chaos.py \
  --attempts 100 \
  --db runs/chaos.db \
  --run-id chaos-r2 \
  --task "send exactly one email" \
  --server-url http://localhost:8000/chat \
  --assert-email-count 1
```

In agent chaos mode the helper starts the deterministic run once, resumes it on later
attempts, kills children with `SIGKILL`, then performs a final resume and checks the
SQLite `sent_emails` count. The older raw-command mode is still available with
`python3 harness/chaos.py -- <command...>`.

## What Works

- Durable finite-state loop with SQLite run state and append-only events.
- Pending tool calls survive in persisted run state and can be recovered from the event log.
- Simulated `send_email` uses SQLite idempotency records keyed by logical email payload.
- Parallel tool batches record every start before any result and isolate failed/hanging tools.
- Explicit false-success checks reject final claims that contradict concrete failed tool results.
- Interrupted partial tool turns terminate legibly without executing incomplete batches.
- Tool boundary for file, Python, HTTP, and email tools.
- Tool results are wrapped as untrusted data before returning to the model.
- Explicit tool registry in `agent/executer.py` exposes `read_file`, `write_file`,
  `run_python`, `http_get`, and `send_email`.
- Workspace path confinement and HTTP host allow-listing.
- `send_email` is blocked after untrusted tool-result data enters the conversation.
- Visible red-team injection cases live in `harness/redteam/`.
- Deterministic context budget using `mockllm/tokenizer.py`.
- JSONL traces and offline replay.
- Retry, backoff, `Retry-After`, and circuit breaker support for mock-server calls.
- YAML-backed runtime defaults.
- Local stdlib mock server for S1-S12 smoke tests.
- Automatic assessment time logging through Makefile targets.

## Known Gaps

- The local mock server is a compatibility shim; final parity still needs the official
  assessment mock server if it differs from these documented behaviours.
- Generic false success claims without a concrete failed target are not rejected yet.
- Successful sibling side effects are not rolled back when another parallel tool fails.
- `run_python` blocks network by monkey-patching `socket` inside isolated Python, not by
  OS-level network namespace/seccomp policy.
- The local chaos helper is lighter than a full `kill -9` durability test.
- Part B is not implemented in this Part A submission.
