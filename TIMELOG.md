# Time Log

| Area | Requirement | Duration | Work completed |
| --- | --- | ---: | --- |
| Repository setup | Part A setup | 0.40h | Created initial repository structure and baseline files. |
| Agent loop | R1 | 1.20h | Implemented model loop, response parsing, tool dispatch, and S01-S06 handling. |
| Durability and side effects | R2 | 1.40h | Added SQLite event log, resume state, and exactly-once email idempotency. |
| Context management | R3 | 0.75h | Added token budgeting and compaction behavior. |
| Tool safety and injection resistance | R4 | 1.00h | Added workspace confinement, HTTP allow-list checks, untrusted tool-result handling, and email gating. |
| Loop and budget control | R5 | 0.65h | Added step ceilings, no-progress detection, retry limits, and graceful termination. |
| Observability and replay | R6 | 0.70h | Added JSONL traces and replay support from recorded state. |
| Evals | R7 | 1.10h | Added S01-S12 eval cases, baseline comparison, and local live scenario checks. |
| Write-up | R8 | 0.45h | Wrote README and DECISIONS documentation. |
| Verification | Part A checks | 0.35h | Ran tests, evals, setup checks, and live scenario checks. |
| Part B | Not started | 0.00h | Part B is intentionally not implemented until released. |


