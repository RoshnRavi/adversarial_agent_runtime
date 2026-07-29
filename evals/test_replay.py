from pathlib import Path
from typing import Any

import pytest

from agent.exceptions import ReplayError
from agent.message import MemoryManager
from agent.run import AgentDatabase, Replayer, TraceReader, TraceWriter
from agent.runtime import AgentRuntime, RuntimeConfig
from agent.validate import RetryEvent


class ScriptedLLM:
    def __init__(
        self,
        responses: list[dict[str, Any]],
        retries_by_call: dict[int, list[Any]] | None = None,
    ) -> None:
        self.responses = list(responses)
        self.retries_by_call = retries_by_call or {}
        self.last_retries: list[Any] = []
        self.call_count = 0

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.call_count += 1
        self.last_retries = list(self.retries_by_call.get(self.call_count, []))
        return self.responses.pop(0)


def test_replay_reads_trace_without_model_server(tmp_path: Path) -> None:
    writer = TraceWriter(tmp_path / "traces")
    writer.log_trace("run-1", "run_started", {"task": "x"}, step=0)
    writer.log_trace("run-1", "run_finished", {"reason": "completed"}, step=1)

    lines = Replayer(TraceReader(tmp_path / "traces")).replay("run-1")

    assert "run_started" in lines[0]
    assert "run_finished" in lines[1]


def test_runtime_writes_complete_decision_trace(tmp_path: Path) -> None:
    config = RuntimeConfig(
        db_path=tmp_path / "agent.db",
        trace_dir=tmp_path / "traces",
        workspace_root=tmp_path / "workspace",
    )
    db = AgentDatabase(config.db_path)
    runtime = AgentRuntime(
        db,
        memory=MemoryManager(config.token_budget),
        tracer=TraceWriter(config.trace_dir),
        llm_client=ScriptedLLM(
            [
                {
                    "content": "I will use a tool",
                    "tool_calls": [
                        "bad-tool-part",
                        {
                            "id": "write-1",
                            "name": "write_file",
                            "arguments": {"path": "r6.txt", "content": "ok"},
                        },
                    ],
                },
                {"content": "done"},
            ],
            retries_by_call={
                2: [RetryEvent(attempt=1, reason="HTTP 429", delay_seconds=0.0)]
            },
        ),
        config=config,
    )

    state = runtime.run_task("trace-run", "write a file")
    events = list(TraceReader(config.trace_dir).read("trace-run"))
    event_types = {event["event_type"] for event in events}

    assert state.status == "FINISHED"
    assert {
        "run_started",
        "llm_request",
        "llm_response",
        "assistant_message",
        "tool_parse_error",
        "tool_calls_parsed",
        "tool_execution_started",
        "tool_result",
        "retry",
        "loop_control",
        "budget_charged",
        "run_finished",
    } <= event_types
    assert all(
        {"run_id", "event_type", "payload", "step", "created_at"} <= set(event)
        for event in events
    )
    db.close()


def test_replay_reconstructs_recorded_decisions_without_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent.executer import ToolExecutor

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("replay must not execute tools")

    monkeypatch.setattr(ToolExecutor, "execute", fail_if_called)
    writer = TraceWriter(tmp_path / "traces")
    writer.log_trace("run-2", "run_started", {"task": "write"}, step=0)
    writer.log_trace("run-2", "llm_response", {"content": "use tool"}, step=1)
    writer.log_trace(
        "run-2",
        "tool_calls_parsed",
        {
            "tool_calls": [
                {
                    "id": "write-1",
                    "name": "write_file",
                    "arguments": {"path": "x.txt", "content": "ok"},
                }
            ]
        },
        step=1,
    )
    writer.log_trace(
        "run-2",
        "tool_execution_started",
        {
            "tool_call": {
                "id": "write-1",
                "name": "write_file",
                "arguments": {"path": "x.txt", "content": "ok"},
            }
        },
        step=1,
    )
    writer.log_trace(
        "run-2",
        "tool_result",
        {
            "tool_call_id": "write-1",
            "tool_name": "write_file",
            "ok": True,
            "result": "wrote x.txt",
            "error": None,
        },
        step=1,
    )
    writer.log_trace("run-2", "run_finished", {"reason": "completed"}, step=2)

    lines = Replayer(TraceReader(tmp_path / "traces")).replay("run-2")

    assert any('llm_response {"content":"use tool"}' in line for line in lines)
    assert any(
        'tool_execution_started name="write_file" id="write-1" '
        'arguments={"content":"ok","path":"x.txt"}' in line
        for line in lines
    )
    assert any(
        'tool_result name="write_file" id="write-1" ok=True result="wrote x.txt" '
        "error=null" in line
        for line in lines
    )
    assert lines[-1] == 'step 2: run_finished reason="completed"'


def test_replay_formats_r1_safety_events(tmp_path: Path) -> None:
    writer = TraceWriter(tmp_path / "traces")
    writer.log_trace(
        "run-r1",
        "model_contradiction",
        {
            "reason": "assistant claimed success for a failed tool result",
            "matched_targets": ["missing.txt"],
        },
        step=2,
    )
    writer.log_trace(
        "run-r1",
        "partial_tool_turn",
        {"expected_tool_call_count": 3, "parsed_tool_call_count": 1},
        step=3,
    )

    lines = Replayer(TraceReader(tmp_path / "traces")).replay("run-r1")

    assert any("model_contradiction" in line and "missing.txt" in line for line in lines)
    assert any("partial_tool_turn" in line and "expected=3 parsed=1" in line for line in lines)


def test_replay_rejects_malformed_jsonl(tmp_path: Path) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    (trace_dir / "bad-run.jsonl").write_text("{not json}\n", encoding="utf-8")

    with pytest.raises(ReplayError, match="Malformed trace line 1"):
        Replayer(TraceReader(trace_dir)).replay("bad-run")


def test_replay_rejects_missing_trace(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="No trace found for run_id missing-run"):
        Replayer(TraceReader(tmp_path / "traces")).replay("missing-run")


def test_agent_user_replay_does_not_construct_runtime(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent import user

    class FailingRuntime:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("replay should not construct AgentRuntime")

    monkeypatch.setattr(user, "AgentRuntime", FailingRuntime)
    monkeypatch.setattr(user, "replay_run", lambda run_id: f"offline replay {run_id}")

    assert user.main(["replay", "cli-run"]) == 0
    assert "offline replay cli-run" in capsys.readouterr().out
