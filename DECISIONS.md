# Architecture Decisions

Runtime bookkeeping lives in `runs/`; task files live in `workspace/`. This keeps user-visible outputs separate from operational evidence used for resume, replay, and exactly-once checks.

The loop is hand-rolled instead of using an agent framework because Part A forbids frameworks and the assessment is mostly about owning the state machine. I kept model transport, tools, memory, persistence, and tracing in separate modules so each boundary can be tested directly.

SQLite is the source of truth. WAL mode is enabled with full synchronous commits, every state transition appends an event, and irreversible `send_email` writes use a logical idempotency key derived from run id plus `to`, `subject`, and `body`. Email insertion and the matching tool-execution record happen in one transaction, so resume can reconcile partial states after `kill -9`. I rejected an in-memory idempotency set because it cannot survive process death.

JSONL traces are the replay source of truth. Replay reads only the recorded transcript and never calls the mock model or local tools, so it can explain decisions without repeating side effects.

Loop control is explicit: the runtime enforces a step ceiling, repeated-tool no-progress limit, per-turn context budget, and cumulative simulated cost budget, and records those decisions in trace events.

Tool batches are two-phase: the runtime records every `tool_execution_started` event before execution, runs non-side-effect tools before side-effect tools, then records results in original call order. This keeps S10 legible, avoids losing sibling results, and prevents `write_file` or `send_email` when an earlier sibling has already failed. Successful `write_file` calls are restored if a later reversible side effect fails; `send_email` stays last because it is irreversible.

Tool results are always treated as data. They are wrapped in an untrusted-data envelope before being shown to the model, and `send_email` is blocked after any tool-result data enters the conversation. File tools are confined to `workspace/`, HTTP uses an allow-list, and `send_email` is only reachable through the tool executor with a runtime-generated idempotency key. I rejected prompt-pattern filtering as the primary defense because hidden red-team content can phrase the same attack many ways.

Compaction keeps protected system messages, the first user message, recent turns, and a bounded durable-facts ledger. The ledger records early user turns and fact-like messages so a turn-3 fact can still be present around turn 40. Final model requests are measured with `mockllm/tokenizer.py` and must fit the 8,000-token hard ceiling.

Final assistant text is checked against failed tool results before the run is marked complete. Concrete target matches and generic success claims are both rejected; explicit failure acknowledgements such as "cannot complete" are allowed. Interrupted partial tool turns terminate without executing incomplete batches because inventing missing tool calls would corrupt the conversation state.

Still unsafe:

1. `run_python` can create workspace files that are not transactionally rolled back.
2. A queued `send_email` cannot be rolled back if a later email in the same batch fails.
3. `run_python` blocks network by patching Python sockets, not by an OS-level sandbox, so native extensions or interpreter escapes would need stronger isolation.

The eval suite covers S01-S12 in `evals/input.yaml` and `evals/cases.py`. Local live S01-S12 checks run through `make live-scenarios`; final parity should still be checked against the official assessment server if it differs.

With two more weeks I would add OS-level isolation and filesystem overlays for `run_python`, preflight email batches before irreversible sends, persist full memory snapshots after compaction, and add integration tests against the official mock server scenarios.
