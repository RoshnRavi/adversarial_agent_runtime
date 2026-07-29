from __future__ import annotations

from pathlib import Path
from typing import Any

import agent.executer as executer_module
from agent.executer import TOOL_REGISTRY, ToolExecutor, list_tools
from agent.response import ToolCall
from agent.run import AgentDatabase
from agent.runtime import RuntimeConfig


def _tool_call(name: str, arguments: dict[str, Any], call_id: str | None = None) -> ToolCall:
    return ToolCall(id=call_id or name, name=name, arguments=arguments)


def _executor(tmp_path: Path, **overrides: Any) -> tuple[ToolExecutor, AgentDatabase, RuntimeConfig]:
    config_values = {
        "db_path": tmp_path / "agent.db",
        "trace_dir": tmp_path / "traces",
        "workspace_root": tmp_path / "workspace",
        "python_timeout_seconds": 1,
        "http_timeout_seconds": 1.0,
        **overrides,
    }
    config = RuntimeConfig(**config_values)
    db = AgentDatabase(config.db_path)
    return ToolExecutor(db, run_id="tool-run", config=config), db, config


def test_list_tools_exposes_required_tool_names() -> None:
    expected = {"read_file", "write_file", "run_python", "http_get", "send_email"}

    assert set(TOOL_REGISTRY) == expected
    assert {tool["name"] for tool in list_tools()} == expected
    assert all("description" in tool and "arguments" in tool for tool in list_tools())


def test_write_and_read_file_are_confined_to_workspace(tmp_path: Path) -> None:
    executor, db, config = _executor(tmp_path)

    write_result = executor.execute(
        _tool_call("write_file", {"path": "nested/report.txt", "content": "hello"}),
        "write-key",
    )
    read_result = executor.execute(
        _tool_call("read_file", {"path": "nested/report.txt"}),
        "read-key",
    )
    escape_result = executor.execute(
        _tool_call("write_file", {"path": "../escape.txt", "content": "bad"}),
        "escape-key",
    )

    assert write_result.ok
    assert read_result.ok
    assert read_result.result == "hello"
    assert (config.workspace_root / "nested/report.txt").read_text(encoding="utf-8") == "hello"
    assert not escape_result.ok
    assert "Path escapes workspace" in (escape_result.error or "")
    db.close()


def test_run_python_returns_output_and_times_out(tmp_path: Path) -> None:
    executor, db, _config = _executor(tmp_path)

    ok_result = executor.execute(
        _tool_call("run_python", {"code": "print('ok')"}),
        "python-ok",
    )
    timeout_result = executor.execute(
        _tool_call("run_python", {"code": "while True: pass"}),
        "python-timeout",
    )

    assert ok_result.ok
    assert ok_result.result["stdout"].strip() == "ok"
    assert ok_result.result["returncode"] == 0
    assert timeout_result.ok
    assert timeout_result.result["returncode"] == -1
    assert "timeout after 1s" in timeout_result.result["stderr"]
    db.close()


def test_run_python_blocks_network_attempts(tmp_path: Path) -> None:
    executor, db, _config = _executor(tmp_path)

    result = executor.execute(
        _tool_call(
            "run_python",
            {
                "code": (
                    "import socket\n"
                    "try:\n"
                    "    socket.create_connection(('127.0.0.1', 80))\n"
                    "except Exception as exc:\n"
                    "    print(type(exc).__name__, exc)\n"
                )
            },
        ),
        "python-network",
    )

    assert result.ok
    assert "network disabled in run_python" in result.result["stdout"]
    db.close()


def test_http_get_allows_configured_localhost(tmp_path: Path, monkeypatch: Any) -> None:
    class FakeHeaders:
        def get_content_charset(self) -> str:
            return "utf-8"

    class FakeResponse:
        headers = FakeHeaders()

        def __enter__(self) -> Any:
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            return None

        def read(self) -> bytes:
            return b"hello from localhost"

    def fake_urlopen(url: str, *, timeout: float) -> FakeResponse:
        assert url == "http://127.0.0.1:8000/"
        assert timeout == 1.0
        return FakeResponse()

    monkeypatch.setattr(executer_module, "urlopen", fake_urlopen)
    executor, db, _config = _executor(tmp_path, http_allow_hosts=("127.0.0.1",))

    result = executor.execute(
        _tool_call("http_get", {"url": "http://127.0.0.1:8000/"}),
        "http-ok",
    )

    assert result.ok
    assert result.result == "hello from localhost"
    db.close()


def test_http_get_rejects_bad_scheme_and_non_allowlisted_host(tmp_path: Path) -> None:
    executor, db, _config = _executor(tmp_path)

    bad_scheme = executor.execute(
        _tool_call("http_get", {"url": "file:///etc/passwd"}),
        "http-bad-scheme",
    )
    bad_host = executor.execute(
        _tool_call("http_get", {"url": "https://example.com"}),
        "http-bad-host",
    )

    assert not bad_scheme.ok
    assert "Unsupported URL scheme" in (bad_scheme.error or "")
    assert not bad_host.ok
    assert "Host is not allow-listed: example.com" in (bad_host.error or "")
    db.close()


def test_send_email_is_idempotent_through_tool_executor(tmp_path: Path) -> None:
    executor, db, _config = _executor(tmp_path)
    call = _tool_call(
        "send_email",
        {"to": "a@example.com", "subject": "Hi", "body": "Once"},
        call_id="email",
    )

    first = executor.execute(call, "email-key")
    second = executor.execute(call, "email-key")

    assert first.ok
    assert second.ok
    assert first.result["idempotency_key"] == second.result["idempotency_key"]
    assert db.count_sent_emails("tool-run") == 1
    db.close()


def test_unknown_tool_returns_failed_tool_result(tmp_path: Path) -> None:
    executor, db, _config = _executor(tmp_path)

    result = executor.execute(_tool_call("missing_tool", {}), "missing-key")

    assert not result.ok
    assert "Unknown tool: missing_tool" in (result.error or "")
    db.close()
