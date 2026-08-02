# Adversarial Agent Runtime

Custom Python runtime for the Part A adversarial-agent assessment. The submitted
implementation lives in `agent/`; Part B is intentionally excluded until it is released.

## Current Status

- Preferred runtime CLI: `python3 -m agent.user ...`
- Installed console script after `make setup`: `agent ...`
- Runtime defaults: `agent/config_agent.yaml`
- Persistence and replay: SQLite plus JSONL traces in `agent/run.py`
- Exposed tools: `read_file`, `write_file`, `run_python`, `http_get`, `send_email`
- Evals: 12 named S01-S12 cases, currently all passing
- Live `run` commands require a mock LLM server listening at the configured
  `server_url`, default `http://localhost:8000/chat`

This checkout includes a small stdlib local server in `mockllm/server.py` for smoke
tests against the documented S01-S12 behaviours. If the official assessment mock
server is provided separately, use that server for final parity checks.

## What Does Not Work / Not Fully Proven

- Official assessment-server parity is not proven without the official server; this repo
  only proves parity against the included local compatibility server.
- `run_python` can still create workspace files that are not transactionally rolled back.
- A queued `send_email` cannot be rolled back if a later email in the same batch fails.
- `run_python` blocks network by monkey-patching Python sockets, not by OS-level
  network namespace or seccomp isolation.
- The local chaos helper is lighter than a full external `kill -9` durability grader.
- Part B is not implemented in this Part A submission.

## Tool Surface

The runtime exposes all tools required by Part A through `agent/executer.py`:

| Tool | Exposed | Runtime behavior |
| --- | --- | --- |
| `read_file(path)` | Yes | Reads UTF-8 text confined to `workspace/`. Path escapes are rejected. |
| `write_file(path, content)` | Yes | Writes UTF-8 text confined to `workspace/`. Path escapes are rejected. |
| `run_python(code)` | Yes | Runs code in a subprocess with wall-clock timeout, memory cap, and best-effort no-network socket monkey-patch. |
| `http_get(url)` | Yes | Fetches only allow-listed hosts from `agent/config_agent.yaml`; refusals are returned as legible tool errors. |
| `send_email(to, subject, body)` | Yes | Simulated but treated as irreversible. Appends exactly-once logical sends to SQLite `sent_emails`. |

## Project Layout

- `agent/`: Part A runtime package.
  - `user.py`: user-facing CLI for `run`, `resume`, and `replay`.
  - `runtime.py`: agent loop, loop control, side-effect ordering, runtime config.
  - `message.py`: message window, untrusted tool-result messages, context compaction.
  - `validate.py`: mock LLM HTTP client, retries, circuit breaker, turn validation.
  - `response.py`: parses final text, tool calls, and malformed response parts.
  - `executer.py`: local tool registry, execution, and safety checks.
  - `run.py`: SQLite state/events, pending tool calls, JSONL trace, replay.
  - `config_agent.yaml`: default paths, limits, server URL, and tool policy.
- `evals/`: pytest tests, eval runner, YAML scripted inputs, case definitions, baseline,
  and `live_scenarios.py` for local S01-S12 live checks.
- `harness/`: chaos helper for interruption testing.
- `mockllm/`: deterministic tokenizer, scenario YAML, and local HTTP mock server.
- `runs/`: runtime databases and JSONL traces.
- `workspace/`: confined filesystem sandbox for task file tools.

## Requirements

- Python 3.10 or newer.
- SQLite through Python's standard `sqlite3` module.
- PyYAML, installed from project dependencies.
- A terminal.
- Network access for `make setup` if dependencies are not already installed.
- Localhost socket binding for live scenario checks.

`make test`, `make eval`, and `replay` do not need a separately started mock server.
`make live-scenarios` starts local mock servers internally. Manual live `run` commands
need a mock server already listening at the configured `server_url`.

## Clean Checkout Verification Runbook

Run these from the repository root.

1. Install the package and dev dependencies.

   ```bash
   make setup
   ```

   Expected: editable install succeeds.

2. Run the unit and regression tests.

   ```bash
   make test
   ```

   Expected: pytest passes.

3. Run the eval suite and baseline comparison.

   ```bash
   make eval
   ```

   Expected: `12` total cases, `12` passing, `0` failing cases,
   pass rate `1.0`, and `baseline diff: none`.

4. Run local live S01-S12 parity checks.

   ```bash
   make live-scenarios
   ```

   Expected: every line from `S01 PASS` through `S12 PASS`, then:

   ```json
   {"failures": 0, "total": 12}
   ```

5. Optional manual live smoke test.

   Terminal A:

   ```bash
   python3 -m mockllm.server --scenario S01 --port 8000
   ```

   Terminal B:

   ```bash
   python3 -m agent.user \
     --db runs/manual-s01.db \
     --server-url http://127.0.0.1:8000/chat \
     run --run-id manual-s01 --task "exercise S01"
   ```

   Expected: stdout shows `run_id=manual-s01` and `status=FINISHED`. Verify:

   ```bash
   test -f runs/manual-s01.db
   test -f runs/traces/manual-s01.jsonl
   test "$(cat workspace/mock_s01.txt)" = "ok"
   ```

   Stop the server in Terminal A with `Ctrl-C`.

## CLI / Make Target Reference

Use `python3 -m agent.user` when running from source. After `make setup`, the `agent`
console script is also available for replay.

### Make Targets

```bash
make setup
make test
make eval
make live-scenarios
make mockllm SCENARIO=S01
make run TASK="write a report"
make clean
```

- `make setup`: installs the package in editable mode with development dependencies.
- `make test`: runs `python3 -m pytest`.
- `make eval`: runs `python3 -m evals.runner` and compares against
  `evals/baseline.json`.
- `make live-scenarios`: starts local mock servers internally and checks S01-S12.
- `make mockllm SCENARIO=S01`: starts the local mock server at
  `http://127.0.0.1:8000/chat`.
- `make run TASK="..."`: runs the agent against the configured server URL.
- `make clean`: removes local test/cache artifacts.

### Mock Server CLI

Start a specific documented scenario:

```bash
python3 -m mockllm.server --scenario S04 --port 8000
```

Useful variants:

```bash
python3 -m mockllm.server --scenario S01
python3 -m mockllm.server --host 127.0.0.1 --port 8774 --scenario S09
curl http://127.0.0.1:8000/health
```

Stop the server with `Ctrl-C`.

### Check a User Prompt

Start the local scripted mock server with S01:

```bash
python3 -m mockllm.server --scenario S01 --port 8000
```

Run the agent with the user prompt as `--task`:

```bash
python3 -m agent.user \
  --db runs/prompt-check.db \
  --server-url http://127.0.0.1:8000/chat \
  run --run-id prompt-check-001 \
  --task "your user prompt here"
```

Replay the recorded run without contacting the model or tools:

```bash
python3 -m agent.user replay prompt-check-001
```

Inspect the raw JSONL trace:

```bash
tail -n 50 runs/traces/prompt-check-001.jsonl
```

Query the SQLite run state:

```bash
sqlite3 runs/prompt-check.db \
  'select status, step_count, termination_reason from runs where run_id="prompt-check-001";'
```

`mockllm.server` behavior is scenario-driven: the prompt is recorded as the task,
but model responses come from the selected scenario, such as `S01` or `S04`.
If port `8000` is busy, use another port such as `8001` and update `--server-url`
to match.

### Runtime CLI

Show help:

```bash
python3 -m agent.user --help
```

Run a live task against a server:

```bash
python3 -m agent.user \
  --db runs/dev.db \
  --server-url http://127.0.0.1:8000/chat \
  run --run-id demo-s01 --task "exercise S01"
```

Resume the same run:

```bash
python3 -m agent.user --db runs/dev.db resume demo-s01
```

Replay without a model server:

```bash
python3 -m agent.user replay demo-s01
agent replay demo-s01
```

Run using default config values:

```bash
python3 -m agent.user run --task "write a file"
python3 -m agent.user run --run-id r2-check --task "send exactly one email"
python3 -m agent.user resume r2-check
```

Global options:

- `--db`: SQLite database path. Default: `runs/agent_events.db`.
- `--server-url`: mock model endpoint. Default: `http://localhost:8000/chat`.
- `run --run-id`: optional deterministic run ID.
- `run --task`: required task string.
- `resume <run_id>`: continue a persisted run.
- `replay <run_id>`: replay recorded trace events without contacting the model or tools.

If a live run fails with:

```text
NetworkFailureError: mock server failed: <urlopen error [Errno 111] Connection refused>
```

then no process is listening at the configured mock server URL. Start the assessment
mock server, run `make mockllm SCENARIO=S01`, or pass the correct `--server-url`.

### Chaos Harness CLI

Start an S09 mock server in Terminal A:

```bash
python3 -m mockllm.server --scenario S09 --port 8000
```

Run the chaos check in Terminal B:

```bash
python3 harness/chaos.py \
  --attempts 100 \
  --db runs/chaos-s09.db \
  --run-id chaos-s09 \
  --task "exercise S09" \
  --server-url http://127.0.0.1:8000/chat \
  --assert-email-count 1
```

Expected: command exits `0`. Optional SQLite verification:

```bash
python3 - <<'PY'
import sqlite3
conn = sqlite3.connect("runs/chaos-s09.db")
count = conn.execute(
    "SELECT COUNT(*) FROM sent_emails WHERE run_id = ?",
    ("chaos-s09",),
).fetchone()[0]
print(count)
PY
```

Expected output: `1`.

The older raw-command chaos mode is still available:

```bash
python3 harness/chaos.py --attempts 10 -- python3 -m agent.user --help
```

## Requirement Verification Matrix

| Requirement | How to verify | Expected result |
| --- | --- | --- |
| R1 - Agent loop survives S01-S12 | Run `make live-scenarios` and `make eval`. | Local S01-S12 all pass; eval cases S01-S12 pass or terminate with the expected legible reason. |
| R2 - Durability and exactly-once side effects | Start S09 mock server, run `python3 harness/chaos.py ... --assert-email-count 1`, then check SQLite `sent_emails`. | Chaos command exits `0`; email count for the run is exactly `1`. |
| R3 - Context budget | Run `make eval`; inspect `agent/config_agent.yaml` for `token_budget: 8000`. | S08 terminates with `ContextLimitExceededError`; requests are measured with `mockllm/tokenizer.py`. |
| R4 - Injection resistance | Run `make eval` and `make test`; inspect S07 result and red-team tests. | S07 finishes with zero sent emails; visible red-team tests pass. |
| R5 - Loop and budget control | Run `make eval`; inspect S04 result and config limits. | S04 terminates with `NoProgressError`; step, no-progress, token, and cost limits are configured. |
| R6 - Observability and replay | Run a live scenario, stop the server, then run `python3 -m agent.user replay <run_id>`. | SQLite events and `runs/traces/<run_id>.jsonl` exist; replay succeeds without a server. |
| R7 - Evals | Run `make eval`. | Baseline diff is none; there are 12 cases covering S01-S12, with at least 4 adversarial cases. |

R8 is documented in `DECISIONS.md`, which stays under the 1,000-word limit and names
remaining unsafe areas and tradeoffs.

## S01-S12 Behavior Verification

Automated local check:

```bash
make live-scenarios
```

Manual pattern for any scenario:

Terminal A:

```bash
python3 -m mockllm.server --scenario SXX --port 8000
```

Terminal B:

```bash
python3 -m agent.user \
  --db runs/live-sXX.db \
  --server-url http://127.0.0.1:8000/chat \
  run --run-id live-sXX --task "exercise SXX"
```

Replace `SXX` with `S01` through `S12`. Use lowercase in file names if preferred, but
keep the mock server scenario uppercase. Stop the server with `Ctrl-C` before switching
to another scenario.

Expected local behavior:

| Scenario | Expected outcome |
| --- | --- |
| S01 | Run finishes and writes `workspace/mock_s01.txt` with content `ok`. |
| S02 | Run finishes after malformed tool arguments are converted into legible tool errors. |
| S03 | Run finishes after unknown-tool and wrong-typed-argument errors are isolated. |
| S04 | Run finishes in bounded time with `NoProgressError: same tool call repeated without progress`. |
| S05 | Run finishes legibly with `NetworkFailureError` after interrupted response handling. |
| S06 | Runtime retries `429`/`529` responses and then finishes. |
| S07 | Prompt-injection file content remains untrusted data; run finishes with zero sent emails. |
| S08 | Oversized context terminates with `ContextLimitExceededError`. |
| S09 | Duplicate tool IDs and duplicate logical email send produce exactly one email row. |
| S10 | Parallel starts are recorded before results; one failing and one hanging tool do not corrupt the run. |
| S11 | False success claim after a failed tool result terminates with `ModelContradictionError`. |
| S12 | Partial interrupted tool-call turn terminates with `PartialToolTurnError` and does not execute incomplete calls. |

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

The loader in `agent/runtime.py` validates and coerces YAML values into
`RuntimeConfig`. Path values remain relative to the process working directory.
`token_budget` limits the compacted context window and is capped at the required
8,000-token ceiling measured by `mockllm/tokenizer.py`; smaller values are useful for
tests. `cost_budget_tokens` is a cumulative simulated run-cost ceiling charged from
deterministic request and response token counts.

## Evals And Tests

Direct commands:

```bash
python3 -m pytest
python3 -m evals.runner
```

`make eval` runs the same eval runner and compares the result with `evals/baseline.json`.
Scripted eval inputs live in `evals/input.yaml`; case names and adversarial flags live
in `evals/cases.py`. The current baseline is:

- 12 total cases
- 12 passing cases
- 0 failing cases
- pass rate 1.0

## Observability And Replay

Every run persists state and events to SQLite through `agent/run.py`. Structured JSONL
traces are written under `runs/traces/` and can be replayed without the mock server:

```bash
python3 -m agent.user replay <actual_run_id>
agent replay <actual_run_id>
```

Replay reconstructs recorded model/tool/final decisions from the JSONL trace only. It
does not contact the model server, execute tools, or repeat side effects.

Useful inspection commands:

```bash
python3 - <<'PY'
import sqlite3
conn = sqlite3.connect("runs/agent_events.db")
for row in conn.execute("SELECT run_id, status, step_count, termination_reason FROM runs"):
    print(row)
PY

tail -n 20 runs/traces/<actual_run_id>.jsonl
```

## Time Log

`TIMELOG.md` is maintained manually. It records how much time went into each Part A
feature or requirement area, plus verification and the Part B status. Update it
directly whenever new implementation or assessment work is added.

## What Works

- Durable finite-state loop with SQLite run state and append-only events.
- Pending tool calls survive in persisted run state and can be recovered from the event log.
- Simulated `send_email` uses SQLite idempotency records keyed by logical email payload.
- Parallel tool batches record every start before any result and isolate failed/hanging tools.
- Explicit false-success checks reject concrete and generic success claims after failed tools.
- Side-effect tools are skipped when an earlier sibling tool fails; reversible writes are restored.
- Interrupted partial tool turns terminate legibly without executing incomplete batches.
- Tool boundary for file, Python, HTTP, and email tools.
- Tool results are wrapped as untrusted data before returning to the model.
- Workspace path confinement and HTTP host allow-listing.
- `send_email` is blocked after untrusted tool-result data enters the conversation.
- Visible red-team injection cases live in `harness/redteam/`.
- Deterministic context budget using `mockllm/tokenizer.py`.
- JSONL traces and offline replay.
- Retry, backoff, `Retry-After`, and circuit breaker support for mock-server calls.
- YAML-backed runtime defaults.
- Local stdlib mock server for S01-S12 smoke tests.
- Automatic assessment time logging through Makefile targets.
