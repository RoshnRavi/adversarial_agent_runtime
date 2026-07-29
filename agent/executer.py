"""Local tool execution for the custom runtime."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

from .exceptions import SecurityViolationError, ToolArgumentError
from .response import ToolCall
from .run import AgentDatabase
from .runtime import DEFAULT_CONFIG, RuntimeConfig, ToolResult

WORKSPACE_ROOT = DEFAULT_CONFIG.workspace_root.resolve()


class ToolError(SecurityViolationError):
    """Raised when a tool request violates runtime policy."""


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    arguments: dict[str, str]
    handler: Callable[[ToolExecutor, dict[str, Any], str], Any] = field(
        repr=False,
        compare=False,
    )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": dict(self.arguments),
        }


def _confined_path(path: str | Path, *, workspace_root: str | Path | None = None) -> Path:
    if not isinstance(path, (str, Path)):
        raise ToolArgumentError("path must be a string")
    root = Path(workspace_root or WORKSPACE_ROOT).resolve()
    candidate = (root / path).resolve()
    if not candidate.is_relative_to(root):
        raise ToolError(f"Path escapes workspace: {path}")
    return candidate


def read_file(path: str | Path, *, workspace_root: str | Path | None = None) -> str:
    return _confined_path(path, workspace_root=workspace_root).read_text(encoding="utf-8")


def write_file(
    path: str | Path,
    content: str,
    *,
    workspace_root: str | Path | None = None,
) -> str:
    if not isinstance(content, str):
        raise ToolArgumentError("content must be a string")
    target = _confined_path(path, workspace_root=workspace_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"wrote {path}"


@dataclass(frozen=True)
class PythonResult:
    stdout: str
    stderr: str
    returncode: int


def _limit_child(memory_mb: int) -> None:
    try:
        import resource

        memory_bytes = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    except Exception:  # noqa: BLE001 - resource limits are best-effort per platform.
        return


def run_python(
    code: str,
    *,
    timeout_seconds: int = DEFAULT_CONFIG.python_timeout_seconds,
    memory_mb: int = DEFAULT_CONFIG.python_memory_mb,
    workspace_root: str | Path | None = None,
) -> PythonResult:
    if not isinstance(code, str):
        raise ToolArgumentError("code must be a string")
    root = Path(workspace_root or WORKSPACE_ROOT).resolve()
    root.mkdir(parents=True, exist_ok=True)
    network_block = """
import socket
def _blocked_socket(*args, **kwargs):
    raise RuntimeError("network disabled in run_python")
socket.socket = _blocked_socket
socket.create_connection = _blocked_socket
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", network_block + "\n" + code],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
            preexec_fn=lambda: _limit_child(memory_mb),
        )
        return PythonResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )
    except subprocess.TimeoutExpired as exc:
        return PythonResult(
            stdout=exc.stdout or "",
            stderr=f"timeout after {timeout_seconds}s",
            returncode=-1,
        )


def http_get(
    url: str,
    *,
    allow_hosts: Iterable[str] = DEFAULT_CONFIG.http_allow_hosts,
    timeout_seconds: float = DEFAULT_CONFIG.http_timeout_seconds,
) -> str:
    if not isinstance(url, str):
        raise ToolArgumentError("url must be a string")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ToolError(f"Unsupported URL scheme: {parsed.scheme or '<missing>'}")
    host = parsed.hostname
    if not host:
        raise ToolError("URL must include a host")
    if host not in set(allow_hosts):
        raise ToolError(f"Host is not allow-listed: {host}")
    with urlopen(url, timeout=timeout_seconds) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset)


def send_email(
    to: str,
    subject: str,
    body: str,
    *,
    db: AgentDatabase | None = None,
    run_id: str = "manual",
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    if not all(isinstance(value, str) for value in (to, subject, body)):
        raise ToolArgumentError("to, subject, and body must be strings")
    key = idempotency_key or f"manual:{run_id}:{to}:{subject}:{body}"
    owned_db = db is None
    database = db or AgentDatabase()
    try:
        return database.send_email_once(
            idempotency_key=key,
            run_id=run_id,
            to=to,
            subject=subject,
            body=body,
        )
    finally:
        if owned_db:
            database.close()


def _execute_read_file(executor: ToolExecutor, args: dict[str, Any], idempotency_key: str) -> str:
    return read_file(args.get("path"), workspace_root=executor.config.workspace_root)


def _execute_write_file(executor: ToolExecutor, args: dict[str, Any], idempotency_key: str) -> str:
    return write_file(
        args.get("path"),
        args.get("content"),
        workspace_root=executor.config.workspace_root,
    )


def _execute_run_python(
    executor: ToolExecutor,
    args: dict[str, Any],
    idempotency_key: str,
) -> dict[str, Any]:
    result = run_python(
        args.get("code"),
        timeout_seconds=executor.config.python_timeout_seconds,
        memory_mb=executor.config.python_memory_mb,
        workspace_root=executor.config.workspace_root,
    )
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


def _execute_http_get(executor: ToolExecutor, args: dict[str, Any], idempotency_key: str) -> str:
    return http_get(
        args.get("url"),
        allow_hosts=executor.config.http_allow_hosts,
        timeout_seconds=executor.config.http_timeout_seconds,
    )


def _execute_send_email(
    executor: ToolExecutor,
    args: dict[str, Any],
    idempotency_key: str,
) -> dict[str, Any]:
    return send_email(
        args.get("to"),
        args.get("subject"),
        args.get("body"),
        db=executor.db,
        run_id=executor.run_id,
        idempotency_key=idempotency_key,
    )


TOOL_REGISTRY: dict[str, ToolDefinition] = {
    "read_file": ToolDefinition(
        name="read_file",
        description="Read a UTF-8 text file confined to the configured workspace.",
        arguments={"path": "Workspace-relative file path to read."},
        handler=_execute_read_file,
    ),
    "write_file": ToolDefinition(
        name="write_file",
        description="Write UTF-8 text to a file confined to the configured workspace.",
        arguments={
            "path": "Workspace-relative file path to write.",
            "content": "Text content to write.",
        },
        handler=_execute_write_file,
    ),
    "run_python": ToolDefinition(
        name="run_python",
        description="Run Python code in a bounded subprocess with no network access.",
        arguments={"code": "Python source code to execute."},
        handler=_execute_run_python,
    ),
    "http_get": ToolDefinition(
        name="http_get",
        description="Fetch an HTTP(S) URL if its host is explicitly allow-listed.",
        arguments={"url": "HTTP or HTTPS URL to fetch."},
        handler=_execute_http_get,
    ),
    "send_email": ToolDefinition(
        name="send_email",
        description="Simulate an irreversible email send with SQLite idempotency.",
        arguments={
            "to": "Recipient email address.",
            "subject": "Email subject.",
            "body": "Email body.",
        },
        handler=_execute_send_email,
    ),
}


def list_tools() -> list[dict[str, Any]]:
    return [definition.to_public_dict() for definition in TOOL_REGISTRY.values()]


class ToolExecutor:
    """Validates and executes model-requested tools through the policy boundary."""

    def __init__(
        self,
        db: AgentDatabase,
        *,
        run_id: str,
        config: RuntimeConfig = DEFAULT_CONFIG,
    ) -> None:
        self.db = db
        self.run_id = run_id
        self.config = config

    def execute(self, call: ToolCall, idempotency_key: str) -> ToolResult:
        existing = self.db.get_tool_execution(idempotency_key)
        if existing is not None:
            return ToolResult(
                tool_call_id=call.id,
                tool_name=call.name,
                ok=existing["status"] == "completed" and not existing["error"],
                result=existing["result"],
                error=existing["error"],
                idempotency_key=idempotency_key,
            )

        try:
            result = self._execute_once(call, idempotency_key)
            self.db.record_tool_execution(
                idempotency_key=idempotency_key,
                run_id=self.run_id,
                tool_call_id=call.id,
                tool_name=call.name,
                arguments=call.arguments,
                result=result,
                status="completed",
            )
            return ToolResult(
                tool_call_id=call.id,
                tool_name=call.name,
                ok=True,
                result=result,
                idempotency_key=idempotency_key,
            )
        except Exception as exc:  # noqa: BLE001 - surface any tool failure as a tool result.
            error = f"{type(exc).__name__}: {exc}"
            self.db.record_tool_execution(
                idempotency_key=idempotency_key,
                run_id=self.run_id,
                tool_call_id=call.id,
                tool_name=call.name,
                arguments=call.arguments,
                error=error,
                status="failed",
            )
            return ToolResult(
                tool_call_id=call.id,
                tool_name=call.name,
                ok=False,
                error=error,
                idempotency_key=idempotency_key,
            )

    def _execute_once(self, call: ToolCall, idempotency_key: str) -> Any:
        args = call.arguments
        if not isinstance(args, dict):
            raise ToolArgumentError("tool arguments must be an object")
        definition = TOOL_REGISTRY.get(call.name)
        if definition is None:
            raise ToolArgumentError(f"Unknown tool: {call.name}")
        return definition.handler(self, args, idempotency_key)
