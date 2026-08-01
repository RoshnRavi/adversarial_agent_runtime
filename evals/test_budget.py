import pytest

import agent.executer as executer_module
from agent.exceptions import ContextLimitExceededError
from agent.message import MemoryManager
from agent.run import AgentDatabase, TraceReader, TraceWriter
from agent.runtime import AgentRuntime, RuntimeConfig


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.last_retries = []

    def call(self, payload):
        if not self.responses:
            return {"content": "done"}
        return self.responses.pop(0)


def test_memory_rejects_single_message_over_budget() -> None:
    memory = MemoryManager(token_budget=5)

    with pytest.raises(ContextLimitExceededError):
        memory.add_user_message("word " * 100)


def _runtime(tmp_path, responses, **config_overrides):
    config = RuntimeConfig(
        db_path=tmp_path / "agent.db",
        trace_dir=tmp_path / "traces",
        workspace_root=tmp_path / "workspace",
        retry_base_delay=0.0,
        **config_overrides,
    )
    db = AgentDatabase(config.db_path)
    runtime = AgentRuntime(
        db,
        memory=MemoryManager(config.token_budget),
        tracer=TraceWriter(config.trace_dir),
        llm_client=ScriptedLLM(responses),
        config=config,
    )
    return runtime, db, config


def test_step_ceiling_terminates_with_legible_reason(tmp_path) -> None:
    responses = [
        {
            "tool_call": {
                "id": f"call-{index}",
                "name": "run_python",
                "arguments": {"code": f"print({index})"},
            }
        }
        for index in range(3)
    ]
    runtime, db, config = _runtime(tmp_path, responses, max_steps=2, no_progress_limit=10)

    state = runtime.run_task("step-limit-run", "keep calling tools")
    traces = list(TraceReader(config.trace_dir).read("step-limit-run"))

    assert state.termination_reason == "MaxStepsReachedError: step limit reached"
    assert any(
        event["event_type"] == "loop_control"
        and event["payload"]["decision"] == "step_limit"
        for event in traces
    )
    db.close()


def test_s4_no_progress_terminates_in_bounded_time_with_trace(tmp_path) -> None:
    repeated_call = {
        "tool_call": {
            "id": "id-can-change",
            "name": "run_python",
            "arguments": {"code": "print(1)"},
        }
    }
    runtime, db, config = _runtime(
        tmp_path,
        [repeated_call, repeated_call, repeated_call],
        max_steps=5,
        no_progress_limit=2,
    )

    state = runtime.run_task("s4-trace-run", "detect infinite loop")
    traces = list(TraceReader(config.trace_dir).read("s4-trace-run"))
    tool_parse_events = [
        event for event in traces if event["event_type"] == "tool_calls_parsed"
    ]
    progress_events = [
        event
        for event in traces
        if event["event_type"] == "loop_control"
        and event["payload"]["decision"] == "progress_check"
    ]

    assert state.step_count == 2
    assert state.termination_reason == "NoProgressError: same tool call repeated without progress"
    assert len(tool_parse_events) == 2
    assert [event["payload"]["repeat_count"] for event in progress_events] == [1, 2]
    assert traces[-1]["event_type"] == "run_finished"
    assert "NoProgressError" in traces[-1]["payload"]["reason"]
    db.close()


def test_s4_repeated_same_tool_id_records_per_step_results_without_rerun(
    tmp_path,
    monkeypatch,
) -> None:
    repeated_call = {
        "tool_call": {
            "id": "same-loop-id",
            "name": "run_python",
            "arguments": {"code": "print(1)"},
        }
    }
    execution_count = 0

    def fake_run_python(*args, **kwargs):
        nonlocal execution_count
        execution_count += 1
        return executer_module.PythonResult(stdout="1\n", stderr="", returncode=0)

    monkeypatch.setattr(executer_module, "run_python", fake_run_python)
    runtime, db, config = _runtime(
        tmp_path,
        [repeated_call, repeated_call, repeated_call],
        max_steps=5,
        no_progress_limit=3,
    )

    state = runtime.run_task("s4-same-id-run", "detect repeated identical call")
    events = db.get_events("s4-same-id-run")
    traces = list(TraceReader(config.trace_dir).read("s4-same-id-run"))
    progress_counts = [
        event.payload["repeat_count"]
        for event in events
        if event.event_type == "loop_control"
        and event.payload.get("decision") == "progress_check"
    ]
    tool_result_steps = [
        event.step for event in events if event.event_type == "tool_result"
    ]
    tool_start_steps = [
        event.step for event in events if event.event_type == "tool_execution_started"
    ]

    assert state.step_count == 3
    assert state.termination_reason == "NoProgressError: same tool call repeated without progress"
    assert progress_counts == [1, 2, 3]
    assert tool_start_steps == [1, 2]
    assert tool_result_steps == [1, 2]
    assert execution_count == 1
    assert traces[-1]["event_type"] == "run_finished"
    assert "NoProgressError" in traces[-1]["payload"]["reason"]
    db.close()


def test_s4_no_progress_state_is_scoped_to_each_run(tmp_path) -> None:
    first_call = {
        "tool_call": {
            "id": "first-run-call",
            "name": "run_python",
            "arguments": {"code": "print(1)"},
        }
    }
    second_call = {
        "tool_call": {
            "id": "second-run-call",
            "name": "run_python",
            "arguments": {"code": "print(1)"},
        }
    }
    runtime, db, config = _runtime(
        tmp_path,
        [first_call, {"content": "done"}, second_call, {"content": "done"}],
        max_steps=5,
        no_progress_limit=2,
    )

    first_state = runtime.run_task("s4-first-run", "run once")
    second_state = runtime.run_task("s4-second-run", "run once again")
    second_progress = [
        event
        for event in TraceReader(config.trace_dir).read("s4-second-run")
        if event["event_type"] == "loop_control"
        and event["payload"]["decision"] == "progress_check"
    ]

    assert first_state.termination_reason == "completed"
    assert second_state.termination_reason == "completed"
    assert [event["payload"]["repeat_count"] for event in second_progress] == [1]
    db.close()


def test_cumulative_cost_budget_terminates_gracefully(tmp_path) -> None:
    runtime, db, config = _runtime(
        tmp_path,
        [{"content": "done"}],
        cost_budget_tokens=1,
    )

    state = runtime.run_task("cost-budget-run", "small task")
    traces = list(TraceReader(config.trace_dir).read("cost-budget-run"))

    assert state.termination_reason == "BudgetExceededError: cost budget exceeded"
    assert any(event["event_type"] == "budget_charged" for event in traces)
    assert traces[-1]["payload"]["reason"] == "BudgetExceededError: cost budget exceeded"
    db.close()


def test_context_token_budget_terminates_gracefully(tmp_path) -> None:
    runtime, db, config = _runtime(
        tmp_path,
        [{"content": "word " * 100}],
        token_budget=5,
    )

    state = runtime.run_task("context-budget-run", "x")
    traces = list(TraceReader(config.trace_dir).read("context-budget-run"))

    assert "ContextLimitExceededError" in (state.termination_reason or "")
    assert traces[-1]["event_type"] == "run_finished"
    assert "ContextLimitExceededError" in traces[-1]["payload"]["reason"]
    db.close()
