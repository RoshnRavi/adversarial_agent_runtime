# adversarial_agent_runtime

Skeleton for an adversarial agent runtime.

## Commands

```bash
make setup
make test
make eval
python3 -m agent.cli run --task "write a report"
python3 -m agent.cli resume <run_id>
python3 -m agent.cli replay <run_id>
```

## Status

This repository currently contains the requested project structure and starter modules. The
implementation is intentionally minimal until the mock LLM server, harness behavior, and eval
contract are finalized.

## Provided-Like Harness Shape

- `mockllm/scenarios/` contains `S01.yml` through `S12.yml` placeholders.
- `mockllm/tokenizer.py` is the deterministic token counter used by agent memory budgeting.
- `harness/chaos.py` repeatedly runs and randomly terminates a command.
- `harness/redteam/` is the mount point for undisclosed adversarial payloads.
# adversarial_agent_runtime
# adversarial_agent_runtime
