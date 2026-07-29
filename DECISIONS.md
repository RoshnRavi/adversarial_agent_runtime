# Architecture Decisions

Runtime bookkeeping lives in `runs/`; task files live in `workspace/`. This keeps user-visible outputs separate from operational evidence used for resume, replay, and exactly-once checks.

The loop is hand-rolled instead of using an agent framework because Part A forbids frameworks and the assessment is mostly about owning the state machine. I kept model transport, tools, memory, persistence, and tracing in separate modules so each boundary can be tested directly.

SQLite is the source of truth. WAL mode is enabled with full synchronous commits, every state transition appends an event, and irreversible `send_email` writes use a logical idempotency key derived from run id plus `to`, `subject`, and `body`. Email insertion and the matching tool-execution record happen in one transaction, so resume can reconcile partial states after `kill -9`. I rejected an in-memory idempotency set because it cannot survive process death.

JSONL traces are the replay source of truth. Replay reads only the recorded transcript and never calls the mock model or local tools, so it can explain decisions without repeating side effects.

Loop control is explicit: the runtime enforces a step ceiling, repeated-tool no-progress limit, per-turn context budget, and cumulative simulated cost budget, and records those decisions in trace events.

Tool batches are two-phase: the runtime records every `tool_execution_started` event before dispatching tools concurrently, then records results in original call order. This keeps S10 legible and avoids silently losing the non-hanging sibling result when another tool times out.

Tool results are always treated as data. They are wrapped in an untrusted-data envelope before being shown to the model, and `send_email` is blocked after any tool-result data enters the conversation. File tools are confined to `workspace/`, HTTP uses an allow-list, and `send_email` is only reachable through the tool executor with a runtime-generated idempotency key. I rejected prompt-pattern filtering as the primary defense because hidden red-team content can phrase the same attack many ways.

Compaction keeps protected system messages, the first user message, recent turns, and a bounded durable-facts ledger. The ledger records early user turns and fact-like messages so a turn-3 fact can still be present around turn 40. Final model requests are measured with `mockllm/tokenizer.py` and must fit the 8,000-token hard ceiling.

Final assistant text is checked against concrete failed tool targets before the run is marked complete. Interrupted partial tool turns terminate without executing incomplete batches because inventing missing tool calls would corrupt the conversation state.

Still unsafe:

1. Generic false success claims without a concrete failed target are not rejected yet.
2. Successful sibling side effects are not rolled back when another parallel tool fails.
3. `run_python` blocks network by patching Python sockets, not by an OS-level sandbox, so native extensions or interpreter escapes would need stronger isolation.

The eval suite intentionally keeps the first two as executed expected failures in `evals/input.yaml` and `evals/cases.py`; a fully green board would hide the current risk instead of documenting it. One passing eval uses the local HTTP mock server; final parity should still be checked against the official assessment server if it differs.

With two more weeks I would add transactional side-effect planning for mixed batches, broaden false-claim verification, persist full memory snapshots after compaction, harden `run_python` with OS-level network isolation, and add integration tests against the official mock server scenarios.
