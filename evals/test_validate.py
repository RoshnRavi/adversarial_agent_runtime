from pathlib import Path
from typing import Any

import pytest

from agent.exceptions import ContextLimitExceededError
from agent.message import MemoryManager, MemoryMessage
from agent.run import AgentDatabase, TraceReader, TraceWriter
from agent.runtime import RuntimeConfig
from agent.validate import AgentValidator, RetryEvent


class ScriptedLLM:
    def __init__(
        self,
        responses: list[dict[str, Any]],
        retries: list[Any] | None = None,
    ) -> None:
        self.responses = list(responses)
        self.last_retries = list(retries or [])

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.responses.pop(0)


def _validator(
    tmp_path: Path,
    run_id: str,
    responses: list[dict[str, Any]],
    *,
    retries: list[Any] | None = None,
    token_budget: int = 8000,
) -> tuple[AgentValidator, AgentDatabase, RuntimeConfig, MemoryManager]:
    config = RuntimeConfig(
        db_path=tmp_path / "agent.db",
        trace_dir=tmp_path / "traces",
        token_budget=token_budget,
    )
    db = AgentDatabase(config.db_path)
    db.create_run(run_id, "validate task")
    memory = MemoryManager(token_budget)
    memory.add_user_message("validate task")
    validator = AgentValidator(
        db,
        TraceWriter(config.trace_dir),
        ScriptedLLM(responses, retries=retries),
    )
    return validator, db, config, memory


def test_validate_turn_records_final_text_response(tmp_path: Path) -> None:
    run_id = "validate-final"
    validator, db, _config, memory = _validator(tmp_path, run_id, [{"content": "done"}])

    validated = validator.validate_turn(run_id, task="validate task", step=1, memory=memory)

    assert validated.assistant_text == "done"
    assert validated.tool_calls == []
    event_types = [event.event_type for event in db.get_events(run_id)]
    assert "llm_request" in event_types
    assert "llm_response" in event_types
    assert "assistant_message" in event_types
    assert db.get_run(run_id).status == "STARTED"
    db.close()


def test_validate_turn_returns_tool_calls_without_finishing_run(tmp_path: Path) -> None:
    run_id = "validate-tool"
    validator, db, _config, memory = _validator(
        tmp_path,
        run_id,
        [
            {
                "tool_call": {
                    "id": "write-1",
                    "name": "write_file",
                    "arguments": {"path": "out.txt", "content": "ok"},
                }
            }
        ],
    )

    validated = validator.validate_turn(run_id, task="validate task", step=1, memory=memory)

    assert len(validated.tool_calls) == 1
    assert validated.tool_calls[0].name == "write_file"
    assert db.get_run(run_id).status == "STARTED"
    event_types = [event.event_type for event in db.get_events(run_id)]
    assert "tool_calls_parsed" not in event_types
    db.close()


def test_validate_turn_records_tool_parse_errors(tmp_path: Path) -> None:
    run_id = "validate-parse-error"
    validator, db, _config, memory = _validator(
        tmp_path,
        run_id,
        [{"content": "done", "tool_calls": ["bad"]}],
    )

    validated = validator.validate_turn(run_id, task="validate task", step=1, memory=memory)

    assert validated.parse_errors == ["tool call 0 is not an object"]
    parse_errors = [
        event.payload["error"]
        for event in db.get_events(run_id)
        if event.event_type == "tool_parse_error"
    ]
    assert parse_errors == ["tool call 0 is not an object"]
    db.close()


def test_validate_turn_writes_retry_trace(tmp_path: Path) -> None:
    run_id = "validate-retry"
    validator, db, config, memory = _validator(
        tmp_path,
        run_id,
        [{"content": "done"}],
        retries=[RetryEvent(attempt=1, reason="HTTP 429", delay_seconds=0.0)],
    )

    validator.validate_turn(run_id, task="validate task", step=1, memory=memory)

    trace_events = list(TraceReader(config.trace_dir).read(run_id))
    assert any(
        event["event_type"] == "retry" and event["payload"]["reason"] == "HTTP 429"
        for event in trace_events
    )
    db.close()


def test_validate_turn_propagates_context_budget_errors(tmp_path: Path) -> None:
    run_id = "validate-budget"
    validator, db, _config, _memory = _validator(
        tmp_path,
        run_id,
        [{"content": "done"}],
    )
    memory = MemoryManager(token_budget=1)
    memory.window.messages.append(MemoryMessage("user", "too many tokens"))

    with pytest.raises(ContextLimitExceededError):
        validator.validate_turn(run_id, task="validate task", step=1, memory=memory)

    db.close()
