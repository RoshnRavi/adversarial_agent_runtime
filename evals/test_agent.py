"""End-to-end runtime tests using scripted model responses."""

from pathlib import Path

from agent.message import MemoryManager, MemoryWindow
from agent.run import AgentDatabase, TraceWriter
from agent.runtime import AgentRuntime, RuntimeConfig
from evals.cases import EVAL_CASES


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.last_retries = []
        self.calls = []

    def call(self, payload):
        self.calls.append(payload)
        return self.responses.pop(0) if self.responses else {"content": "done"}


def test_memory_window_accepts_messages() -> None:
    memory = MemoryWindow(token_budget=100)
    memory.add("user", "hello")
    assert memory.messages[0].content == "hello"


def test_loop_executes_tool_and_records_trace(tmp_path: Path) -> None:
    config = RuntimeConfig(db_path=tmp_path / "agent.db", trace_dir=tmp_path / "traces")
    db = AgentDatabase(config.db_path)
    runtime = AgentRuntime(
        db,
        memory=MemoryManager(config.token_budget),
        tracer=TraceWriter(config.trace_dir),
        llm_client=ScriptedLLM(
            [
                {
                    "tool_call": {
                        "id": "write-1",
                        "name": "write_file",
                        "arguments": {"path": "unit/write.txt", "content": "ok"},
                    }
                },
                {"content": "done"},
            ]
        ),
        config=config,
    )

    state = runtime.run_task("unit-run", "write a file")

    assert state.status == "FINISHED"
    assert (Path("workspace") / "unit/write.txt").read_text(encoding="utf-8") == "ok"
    event_types = [event.event_type for event in db.get_events("unit-run")]
    assert "tool_result" in event_types
    assert (tmp_path / "traces/unit-run.jsonl").exists()
    db.close()


def test_parallel_tool_results_keep_each_tool_name_in_memory(tmp_path: Path) -> None:
    config = RuntimeConfig(
        db_path=tmp_path / "agent.db",
        trace_dir=tmp_path / "traces",
        workspace_root=tmp_path / "workspace",
    )
    llm = ScriptedLLM(
        [
            {
                "tool_calls": [
                    {
                        "id": "write-1",
                        "name": "write_file",
                        "arguments": {"path": "memory/write.txt", "content": "ok"},
                    },
                    {
                        "id": "read-1",
                        "name": "read_file",
                        "arguments": {"path": "missing-memory.txt"},
                    },
                ]
            },
            {"content": "done"},
        ]
    )
    db = AgentDatabase(config.db_path)
    runtime = AgentRuntime(
        db,
        memory=MemoryManager(config.token_budget),
        tracer=TraceWriter(config.trace_dir),
        llm_client=llm,
        config=config,
    )

    state = runtime.run_task("memory-tool-name-run", "run two tools")
    second_request = llm.calls[1]
    rendered_messages = "\n".join(message["content"] for message in second_request["messages"])

    assert state.status == "FINISHED"
    assert '"tool":"write_file"' in rendered_messages
    assert '"tool":"read_file"' in rendered_messages
    db.close()


def test_generic_success_after_failed_tool_is_rejected(tmp_path: Path) -> None:
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
                    "tool_call": {
                        "id": "missing",
                        "name": "read_file",
                        "arguments": {"path": "missing-generic.txt"},
                    }
                },
                {"content": "Everything succeeded."},
            ]
        ),
        config=config,
    )

    state = runtime.run_task("generic-false-success-run", "read a file")

    assert "ModelContradictionError" in (state.termination_reason or "")
    db.close()


def test_mixed_batch_skips_write_after_sibling_failure(tmp_path: Path) -> None:
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
                    "tool_calls": [
                        {
                            "id": "write",
                            "name": "write_file",
                            "arguments": {
                                "path": "mixed/side-effect.txt",
                                "content": "should not persist",
                            },
                        },
                        {
                            "id": "fail",
                            "name": "read_file",
                            "arguments": {"path": "missing-mixed.txt"},
                        },
                    ]
                },
                {"content": "done"},
            ]
        ),
        config=config,
    )

    state = runtime.run_task("mixed-batch-run", "avoid partial side effects")
    tool_results = [
        event.payload for event in db.get_events("mixed-batch-run")
        if event.event_type == "tool_result"
    ]

    assert state.status == "FINISHED"
    assert not (config.workspace_root / "mixed/side-effect.txt").exists()
    assert any(
        result["tool_name"] == "read_file" and result["ok"] is False
        for result in tool_results
    )
    assert any(
        result["tool_name"] == "write_file"
        and result["ok"] is False
        and "ToolSkippedError" in (result["error"] or "")
        for result in tool_results
    )
    db.close()


def test_write_rolls_back_after_later_send_email_failure(tmp_path: Path) -> None:
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
                    "tool_calls": [
                        {
                            "id": "write",
                            "name": "write_file",
                            "arguments": {
                                "path": "mixed/email-failure.txt",
                                "content": "should be rolled back",
                            },
                        },
                        {
                            "id": "email",
                            "name": "send_email",
                            "arguments": {
                                "to": "user@example.com",
                                "subject": "subject",
                                "body": 42,
                            },
                        },
                    ]
                },
                {"content": "done"},
            ]
        ),
        config=config,
    )

    state = runtime.run_task("write-email-rollback-run", "avoid partial side effects")
    tool_results = [
        event.payload for event in db.get_events("write-email-rollback-run")
        if event.event_type == "tool_result"
    ]

    assert state.status == "FINISHED"
    assert not (config.workspace_root / "mixed/email-failure.txt").exists()
    assert any(
        result["tool_name"] == "write_file"
        and result["ok"] is False
        and "ToolRolledBackError" in (result["error"] or "")
        for result in tool_results
    )
    assert any(
        result["tool_name"] == "send_email" and result["ok"] is False
        for result in tool_results
    )
    db.close()


def test_eval_suite_has_required_case_count() -> None:
    assert len(EVAL_CASES) >= 12
    assert sum(1 for case in EVAL_CASES if case.adversarial) >= 4
    assert sum(1 for case in EVAL_CASES if case.expected_failure) >= 2
