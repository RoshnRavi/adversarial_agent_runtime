# Architecture Decisions

## Runtime Artifacts

Runtime bookkeeping lives in `runs/`, not `workspace/`.

`workspace/` is reserved for task files created or modified on behalf of the agent user. `runs/`
contains operational records such as the SQLite event log and JSONL traces. Keeping these separate
makes it clear which files are user-facing task outputs and which files are agent-runtime evidence
needed for debugging, resume, replay, and exactly-once bookkeeping.

## Tool Boundary

Agent tools are isolated in `agent/tools.py` so sandboxing, path confinement, timeouts, memory
limits, HTTP allow-lists, and irreversible-write handling can be audited in one place.
