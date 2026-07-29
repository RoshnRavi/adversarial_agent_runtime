from pathlib import Path

from agent.executer import ToolError, http_get, write_file
from agent.message import MemoryManager
from agent.run import AgentDatabase, TraceWriter
from agent.runtime import AgentRuntime, RuntimeConfig


class RepeatingLLM:
    def __init__(self) -> None:
        self.last_retries = []

    def call(self, payload):
        return {
            "tool_call": {
                "id": "changes-each-turn",
                "name": "run_python",
                "arguments": {"code": "print(1)"},
            }
        }


def test_write_file_rejects_workspace_escape() -> None:
    try:
        write_file("../escape.txt", "no")
    except ToolError:
        return

    raise AssertionError("workspace escape should be rejected")


def test_http_get_rejects_non_allowlisted_host() -> None:
    try:
        http_get("https://example.com")
    except ToolError:
        return

    raise AssertionError("non allow-listed host should be rejected")


def test_repeated_tool_call_terminates_with_reason(tmp_path: Path) -> None:
    config = RuntimeConfig(
        db_path=tmp_path / "agent.db",
        trace_dir=tmp_path / "traces",
        no_progress_limit=2,
        max_steps=5,
    )
    db = AgentDatabase(config.db_path)
    runtime = AgentRuntime(
        db,
        memory=MemoryManager(config.token_budget),
        tracer=TraceWriter(config.trace_dir),
        llm_client=RepeatingLLM(),
        config=config,
    )

    state = runtime.run_task("loop-run", "repeat")

    assert "NoProgressError" in (state.termination_reason or "")
    db.close()
