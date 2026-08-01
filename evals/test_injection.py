from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agent.message import MemoryManager
from agent.run import AgentDatabase, TraceWriter
from agent.runtime import AgentRuntime, RuntimeConfig

REDTEAM_CASES = Path("harness/redteam/injection_cases.yaml")


class ScriptedLLM:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.last_retries: list[Any] = []

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.responses:
            return {"content": "done"}
        return self.responses.pop(0)


def _load_case(case_id: str) -> dict[str, Any]:
    loaded = yaml.safe_load(REDTEAM_CASES.read_text(encoding="utf-8"))
    return loaded["cases"][case_id]


def _runtime_for_case(
    tmp_path: Path,
    case: dict[str, Any],
    followup_response: dict[str, Any],
) -> tuple[AgentDatabase, RuntimeConfig]:
    config = RuntimeConfig(
        db_path=tmp_path / "agent.db",
        trace_dir=tmp_path / "traces",
        workspace_root=tmp_path / "workspace",
        python_timeout_seconds=1,
        http_timeout_seconds=1.0,
    )
    config.workspace_root.mkdir(parents=True)
    (config.workspace_root / case["file_path"]).write_text(case["file_content"], encoding="utf-8")
    db = AgentDatabase(config.db_path)
    runtime = AgentRuntime(
        db,
        memory=MemoryManager(config.token_budget),
        tracer=TraceWriter(config.trace_dir),
        llm_client=ScriptedLLM(
            [
                {
                    "tool_call": {
                        "id": "read-injected",
                        "name": "read_file",
                        "arguments": {"path": case["file_path"]},
                    }
                },
                followup_response,
                {"content": "done"},
            ]
        ),
        config=config,
    )

    runtime.run_task(f"redteam-{case['file_path']}", "read the file and summarize it")
    return db, config


def _tool_results(db: AgentDatabase, run_id: str) -> list[dict[str, Any]]:
    return [
        event.payload
        for event in db.get_events(run_id)
        if event.event_type == "tool_result"
    ]


def test_tool_result_injection_cannot_trigger_send_email(tmp_path: Path) -> None:
    case = _load_case("send_email_from_tool_result")
    db, _config = _runtime_for_case(
        tmp_path,
        case,
        {"tool_call": case["blocked_tool_call"]},
    )
    run_id = f"redteam-{case['file_path']}"
    results = _tool_results(db, run_id)

    assert db.count_sent_emails(run_id) == 0
    assert any(
        result["tool_name"] == "send_email"
        and result["ok"] is False
        and "blocked after untrusted tool results" in (result["error"] or "")
        for result in results
    )
    db.close()


def test_tool_result_injection_cannot_write_outside_workspace(tmp_path: Path) -> None:
    case = _load_case("workspace_escape_from_tool_result")
    db, config = _runtime_for_case(
        tmp_path,
        case,
        {"tool_call": case["blocked_tool_call"]},
    )
    run_id = f"redteam-{case['file_path']}"
    results = _tool_results(db, run_id)

    assert not (config.workspace_root.parent / "escape.txt").exists()
    assert any(
        result["tool_name"] == "write_file"
        and result["ok"] is False
        and "Path escapes workspace" in (result["error"] or "")
        for result in results
    )
    db.close()


def test_tool_result_injection_cannot_bypass_http_allowlist(tmp_path: Path) -> None:
    case = _load_case("http_allowlist_bypass_from_tool_result")
    db, _config = _runtime_for_case(
        tmp_path,
        case,
        {"tool_calls": case["blocked_tool_calls"]},
    )
    run_id = f"redteam-{case['file_path']}"
    http_results = [
        result for result in _tool_results(db, run_id) if result["tool_name"] == "http_get"
    ]

    assert len(http_results) == 4
    assert all(result["ok"] is False for result in http_results)
    assert any("Unsupported URL scheme" in (result["error"] or "") for result in http_results)
    assert any("Host is not allow-listed: example.com" in (result["error"] or "") for result in http_results)
    assert any("Host is not allow-listed: 169.254.169.254" in (result["error"] or "") for result in http_results)
    db.close()


def test_tool_result_injection_cannot_enable_python_network(tmp_path: Path) -> None:
    case = _load_case("python_network_from_tool_result")
    db, _config = _runtime_for_case(
        tmp_path,
        case,
        {"tool_call": case["blocked_tool_call"]},
    )
    run_id = f"redteam-{case['file_path']}"
    results = _tool_results(db, run_id)

    assert any(
        result["tool_name"] == "run_python"
        and result["ok"] is True
        and "network disabled in run_python" in result["result"]["stdout"]
        for result in results
    )
    db.close()
