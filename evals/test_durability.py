"""Tests for resume, idempotency, and exactly-once side-effect durability."""

from pathlib import Path

from agent.response import ToolCall
from agent.run import AgentDatabase, TraceWriter
from agent.runtime import AgentRuntime, RuntimeConfig


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.last_retries = []
        self.calls = []

    def call(self, payload):
        self.calls.append(payload)
        if not self.responses:
            return {"content": "done"}
        return self.responses.pop(0)


class BombLLM:
    def __init__(self):
        self.last_retries = []

    def call(self, payload):
        raise AssertionError(f"model should not be called: {payload}")


def _config(tmp_path: Path, **overrides) -> RuntimeConfig:
    return RuntimeConfig(
        db_path=tmp_path / "agent.db",
        trace_dir=tmp_path / "traces",
        workspace_root=tmp_path / "workspace",
        retry_base_delay=0.0,
        **overrides,
    )


def _runtime(tmp_path: Path, responses=None, *, llm_client=None, **config_overrides):
    config = _config(tmp_path, **config_overrides)
    db = AgentDatabase(config.db_path)
    runtime = AgentRuntime(
        db,
        memory=None,
        tracer=TraceWriter(config.trace_dir),
        llm_client=llm_client or ScriptedLLM(responses or []),
        config=config,
    )
    return runtime, db, config


def _email_call(call_id: str = "email") -> ToolCall:
    return ToolCall(
        id=call_id,
        name="send_email",
        arguments={"to": "a@example.com", "subject": "Hi", "body": "Once"},
    )


def _loop_call(call_id: str = "loop") -> ToolCall:
    return ToolCall(
        id=call_id,
        name="run_python",
        arguments={"code": "print(1)"},
    )


def _loop_response(call: ToolCall) -> dict:
    return {
        "tool_call": {
            "id": call.id,
            "name": call.name,
            "arguments": call.arguments,
        }
    }


def _pending_payload(call: ToolCall) -> list[dict]:
    return [
        {
            "id": call.id,
            "name": call.name,
            "arguments": call.arguments,
            "raw_arguments": call.raw_arguments,
        }
    ]


def test_sqlite_wal_is_enabled(tmp_path: Path) -> None:
    db = AgentDatabase(tmp_path / "agent.db")
    row = db.conn.execute("PRAGMA journal_mode").fetchone()
    assert row[0].lower() == "wal"
    db.close()


def test_send_email_is_exactly_once_by_idempotency_key(tmp_path: Path) -> None:
    db = AgentDatabase(tmp_path / "agent.db")
    first = db.send_email_once(
        idempotency_key="email-key",
        run_id="run-1",
        to="a@example.com",
        subject="Hello",
        body="Body",
    )
    second = db.send_email_once(
        idempotency_key="email-key",
        run_id="run-1",
        to="a@example.com",
        subject="Hello",
        body="Body",
    )

    assert first["idempotency_key"] == second["idempotency_key"]
    assert db.count_sent_emails("run-1") == 1
    db.close()


def test_send_email_logical_payload_is_once_across_different_tool_ids(tmp_path: Path) -> None:
    runtime, db, _config = _runtime(
        tmp_path,
        [
            {
                "tool_calls": [
                    {
                        "id": "email-a",
                        "name": "send_email",
                        "arguments": _email_call().arguments,
                    },
                    {
                        "id": "email-b",
                        "name": "send_email",
                        "arguments": _email_call().arguments,
                    },
                ]
            },
            {"content": "done"},
        ],
    )

    state = runtime.run_task("logical-email-run", "send exactly one email")
    tool_results = [
        event for event in db.get_events("logical-email-run") if event.event_type == "tool_result"
    ]

    assert state.status == "FINISHED"
    assert db.count_sent_emails("logical-email-run") == 1
    assert len(tool_results) == 1
    db.close()


def test_resume_persisted_pending_tool_call_sends_email_once(tmp_path: Path) -> None:
    runtime, db, _config = _runtime(tmp_path, [{"content": "done"}])
    call = _email_call()
    db.create_run("pending-run", "send exactly one email")
    db.update_run(
        "pending-run",
        status="RUNNING_TOOLS",
        step_count=1,
        pending_tool_calls=_pending_payload(call),
    )

    state = runtime.resume("pending-run")
    resumed_again = runtime.resume("pending-run")

    assert state.status == "FINISHED"
    assert resumed_again.status == "FINISHED"
    assert db.count_sent_emails("pending-run") == 1
    db.close()


def test_resume_recovers_pending_calls_from_tool_calls_parsed_event(tmp_path: Path) -> None:
    runtime, db, _config = _runtime(tmp_path, [{"content": "done"}])
    call = _email_call()
    db.create_run("parsed-gap-run", "send exactly one email")
    db.update_run("parsed-gap-run", status="CALLING_LLM", step_count=1)
    db.append_event(
        "parsed-gap-run",
        "tool_calls_parsed",
        {"tool_calls": _pending_payload(call)},
        step=1,
        idempotency_key="parsed-gap-run:tool_calls_parsed:1",
    )

    state = runtime.resume("parsed-gap-run")

    assert state.status == "FINISHED"
    assert db.count_sent_emails("parsed-gap-run") == 1
    db.close()


def test_resume_reconciles_existing_email_without_tool_execution_record(tmp_path: Path) -> None:
    runtime, db, _config = _runtime(tmp_path, [{"content": "done"}])
    call = _email_call()
    db.create_run("email-gap-run", "send exactly one email")
    db.update_run(
        "email-gap-run",
        status="RUNNING_TOOLS",
        step_count=1,
        pending_tool_calls=_pending_payload(call),
    )
    idempotency_key = runtime._idempotency_key("email-gap-run", call)
    db.send_email_once(
        idempotency_key=idempotency_key,
        run_id="email-gap-run",
        to=call.arguments["to"],
        subject=call.arguments["subject"],
        body=call.arguments["body"],
    )

    state = runtime.resume("email-gap-run")
    execution = db.get_tool_execution(idempotency_key)

    assert state.status == "FINISHED"
    assert db.count_sent_emails("email-gap-run") == 1
    assert execution is not None
    assert execution["status"] == "completed"
    db.close()


def test_resume_does_not_duplicate_recorded_tool_result_event(tmp_path: Path) -> None:
    runtime, db, _config = _runtime(tmp_path, [{"content": "done"}])
    call = _email_call()
    db.create_run("tool-result-gap-run", "send exactly one email")
    db.update_run(
        "tool-result-gap-run",
        status="RUNNING_TOOLS",
        step_count=1,
        pending_tool_calls=_pending_payload(call),
    )
    idempotency_key = runtime._idempotency_key("tool-result-gap-run", call)
    result = db.send_email_tool_once(
        idempotency_key=idempotency_key,
        run_id="tool-result-gap-run",
        tool_call_id=call.id,
        arguments=call.arguments,
        to=call.arguments["to"],
        subject=call.arguments["subject"],
        body=call.arguments["body"],
    )
    payload = {
        "tool_call_id": call.id,
        "tool_name": call.name,
        "ok": True,
        "result": result,
        "error": None,
        "idempotency_key": idempotency_key,
    }
    db.append_event(
        "tool-result-gap-run",
        "tool_result",
        payload,
        step=1,
        idempotency_key=f"result:{idempotency_key}",
    )

    state = runtime.resume("tool-result-gap-run")
    tool_results = [
        event
        for event in db.get_events("tool-result-gap-run")
        if event.event_type == "tool_result"
    ]

    assert state.status == "FINISHED"
    assert len(tool_results) == 1
    assert db.count_sent_emails("tool-result-gap-run") == 1
    db.close()


def test_resume_uses_recorded_final_llm_response_without_calling_model(tmp_path: Path) -> None:
    runtime, db, _config = _runtime(tmp_path, llm_client=BombLLM())
    db.create_run("recorded-final-run", "finish")
    db.update_run("recorded-final-run", status="CALLING_LLM", step_count=1)
    db.append_event(
        "recorded-final-run",
        "llm_request",
        {"message_count": 1, "tokens": 3},
        step=1,
        idempotency_key="recorded-final-run:llm_request:1",
    )
    db.append_event(
        "recorded-final-run",
        "llm_response",
        {"content": "done"},
        step=1,
        idempotency_key="recorded-final-run:llm_response:1",
    )

    state = runtime.resume("recorded-final-run")

    assert state.status == "FINISHED"
    assert state.step_count == 1
    assert state.termination_reason == "completed"
    db.close()


def test_resume_retries_same_step_after_in_flight_llm_request(tmp_path: Path) -> None:
    client = ScriptedLLM([{"content": "done"}])
    runtime, db, _config = _runtime(tmp_path, llm_client=client)
    db.create_run("in-flight-run", "finish")
    db.update_run("in-flight-run", status="CALLING_LLM", step_count=1)
    db.append_event(
        "in-flight-run",
        "llm_request",
        {"message_count": 1, "tokens": 3},
        step=1,
        idempotency_key="in-flight-run:llm_request:1",
    )

    state = runtime.resume("in-flight-run")

    assert state.status == "FINISHED"
    assert state.step_count == 1
    assert client.calls[0]["step"] == 1
    db.close()


def test_resume_restores_s4_repeat_count_before_executing_next_repeat(
    tmp_path: Path,
) -> None:
    first_call = _loop_call("loop-1")
    second_call = _loop_call("loop-2")
    runtime, db, _config = _runtime(
        tmp_path,
        [_loop_response(second_call), {"content": "done"}],
        no_progress_limit=2,
    )
    db.create_run("resume-s4-run", "detect loop")
    db.update_run("resume-s4-run", status="CALLING_LLM", step_count=1)
    first_idempotency_key = runtime._idempotency_key("resume-s4-run", first_call)
    db.append_event(
        "resume-s4-run",
        "llm_response",
        _loop_response(first_call),
        step=1,
        idempotency_key="resume-s4-run:llm_response:1",
    )
    db.append_event(
        "resume-s4-run",
        "tool_calls_parsed",
        {"tool_calls": _pending_payload(first_call)},
        step=1,
        idempotency_key="resume-s4-run:tool_calls_parsed:1",
    )
    db.append_event(
        "resume-s4-run",
        "loop_control",
        {
            "decision": "progress_check",
            "tool_digest": runtime._tool_progress_digest([first_call]),
            "tool_count": 1,
            "repeat_count": 1,
            "no_progress_limit": 2,
        },
        step=1,
        idempotency_key="resume-s4-run:progress:1",
    )
    db.append_event(
        "resume-s4-run",
        "tool_result",
        {
            "tool_call_id": first_call.id,
            "tool_name": first_call.name,
            "ok": True,
            "result": "1\n",
            "error": None,
            "idempotency_key": first_idempotency_key,
        },
        step=1,
        idempotency_key=f"result:{first_idempotency_key}",
    )

    state = runtime.resume("resume-s4-run")
    events = db.get_events("resume-s4-run")
    progress_counts = [
        event.payload["repeat_count"]
        for event in events
        if event.event_type == "loop_control"
        and event.payload.get("decision") == "progress_check"
    ]

    assert state.termination_reason == "NoProgressError: same tool call repeated without progress"
    assert progress_counts == [1, 2]
    assert not any(
        event.event_type == "tool_execution_started" and event.step == 2
        for event in events
    )
    db.close()


def test_resume_finishes_terminal_s4_progress_without_executing_pending_call(
    tmp_path: Path,
) -> None:
    call = _loop_call("terminal-loop")
    runtime, db, _config = _runtime(
        tmp_path,
        llm_client=BombLLM(),
        no_progress_limit=2,
    )
    db.create_run("terminal-s4-run", "detect loop")
    db.update_run(
        "terminal-s4-run",
        status="RUNNING_TOOLS",
        step_count=2,
        pending_tool_calls=_pending_payload(call),
    )
    db.append_event(
        "terminal-s4-run",
        "tool_calls_parsed",
        {"tool_calls": _pending_payload(call)},
        step=2,
        idempotency_key="terminal-s4-run:tool_calls_parsed:2",
    )
    db.append_event(
        "terminal-s4-run",
        "loop_control",
        {
            "decision": "progress_check",
            "tool_digest": runtime._tool_progress_digest([call]),
            "tool_count": 1,
            "repeat_count": 2,
            "no_progress_limit": 2,
        },
        step=2,
        idempotency_key="terminal-s4-run:progress:2",
    )

    state = runtime.resume("terminal-s4-run")
    events = db.get_events("terminal-s4-run")

    assert state.status == "FINISHED"
    assert state.pending_tool_calls == []
    assert state.termination_reason == "NoProgressError: same tool call repeated without progress"
    assert not any(event.event_type == "tool_execution_started" for event in events)
    db.close()
