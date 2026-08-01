"""Run live local mock-server checks for S01-S12."""

from __future__ import annotations

import json
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent.message import MemoryManager
from agent.run import AgentDatabase, TraceWriter
from agent.runtime import AgentRuntime, RuntimeConfig
from mockllm.server import build_server


SCENARIOS = [f"S{index:02d}" for index in range(1, 13)]


def main() -> int:
    failures = 0
    for scenario in SCENARIOS:
        passed, detail = run_scenario(scenario)
        status = "PASS" if passed else "FAIL"
        print(f"{scenario} {status}" + (f" - {detail}" if detail else ""))
        failures += int(not passed)
    print(json.dumps({"total": len(SCENARIOS), "failures": failures}, sort_keys=True))
    return 1 if failures else 0


def run_scenario(scenario: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        server = build_server("127.0.0.1", 0, scenario=scenario)
        host, port = server.server_address
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            config = RuntimeConfig(
                db_path=tmp / "agent.db",
                trace_dir=tmp / "traces",
                workspace_root=tmp / "workspace",
                server_url=f"http://{host}:{port}/chat",
                retry_base_delay=0.0,
                python_timeout_seconds=1,
            )
            db = AgentDatabase(config.db_path)
            try:
                runtime = AgentRuntime(
                    db,
                    memory=MemoryManager(config.token_budget),
                    tracer=TraceWriter(config.trace_dir),
                    config=config,
                )
                run_id = f"live-{scenario.lower()}"
                state = runtime.run_task(run_id, f"exercise {scenario}")
                return _check_scenario(scenario, state, db, config)
            finally:
                db.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


def _check_scenario(
    scenario: str,
    state: Any,
    db: AgentDatabase,
    config: RuntimeConfig,
) -> tuple[bool, str]:
    checks: dict[str, Callable[[], tuple[bool, str]]] = {
        "S01": lambda: _expect(
            state.status == "FINISHED"
            and _read_if_exists(config.workspace_root / "mock_s01.txt") == "ok",
            state.termination_reason or "",
        ),
        "S02": lambda: _expect_finished(state),
        "S03": lambda: _expect_finished(state),
        "S04": lambda: _expect_reason(state, "NoProgressError"),
        "S05": lambda: _expect_reason(state, "NetworkFailureError"),
        "S06": lambda: _expect_finished(state),
        "S07": lambda: _expect(
            state.status == "FINISHED" and db.count_sent_emails(state.run_id) == 0,
            state.termination_reason or "unexpected email count",
        ),
        "S08": lambda: _expect_reason(state, "ContextLimitExceededError"),
        "S09": lambda: _expect(
            state.status == "FINISHED" and db.count_sent_emails(state.run_id) == 1,
            state.termination_reason or "email count was not one",
        ),
        "S10": lambda: _expect(_parallel_starts_precede_results(db, state.run_id), "bad event order"),
        "S11": lambda: _expect_reason(state, "ModelContradictionError"),
        "S12": lambda: _expect_reason(state, "PartialToolTurnError"),
    }
    return checks[scenario]()


def _expect_finished(state: Any) -> tuple[bool, str]:
    return _expect(state.status == "FINISHED", state.termination_reason or "")


def _expect_reason(state: Any, marker: str) -> tuple[bool, str]:
    reason = state.termination_reason or ""
    return _expect(marker in reason, reason)


def _expect(condition: bool, detail: str) -> tuple[bool, str]:
    return condition, "" if condition else detail


def _read_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _parallel_starts_precede_results(db: AgentDatabase, run_id: str) -> bool:
    events = db.get_events(run_id)
    start_indexes = [
        index for index, event in enumerate(events)
        if event.event_type == "tool_execution_started"
    ]
    result_indexes = [
        index for index, event in enumerate(events)
        if event.event_type == "tool_result"
    ]
    return bool(start_indexes and result_indexes) and max(start_indexes) < min(result_indexes)


if __name__ == "__main__":
    raise SystemExit(main())
