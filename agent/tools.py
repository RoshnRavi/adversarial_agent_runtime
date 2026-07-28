"""Auditable tool implementations for the custom runtime."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse
from urllib.request import urlopen

from .config import DEFAULT_CONFIG, RuntimeConfig
from .database import AgentDatabase
from .dto import ToolCall, ToolResult
from .exceptions import SecurityViolationError, ToolArgumentError


WORKSPACE_ROOT = DEFAULT_CONFIG.workspace_root.resolve()


class ToolError(SecurityViolationError):
    """Raised when a tool request violates runtime policy."""


def _confined_path(path: str | Path, *, workspace_root: Path = WORKSPACE_ROOT) -> Path:
    if not isinstance(path, (str, Path)):
        raise ToolArgumentError("path must be a string")
    root = workspace_root.resolve()
    candidate = (root / path).resolve()
    if not candidate.is_relative_to(root):
        raise ToolError(f"Path escapes workspace: {path}")
    return candidate


def read_file(path: str | Path) -> str:
    return _confined_path(path).read_text(encoding="utf-8")


def write_file(path: str | Path, content: str) -> str:
    if not isinstance(content, str):
        raise ToolArgumentError("content must be a string")
    target = _confined_path(path)
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
    except Exception:
        return


def run_python(
    code: str,
    *,
    timeout_seconds: int = DEFAULT_CONFIG.python_timeout_seconds,
    memory_mb: int = DEFAULT_CONFIG.python_memory_mb,
) -> PythonResult:
    if not isinstance(code, str):
        raise ToolArgumentError("code must be a string")
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
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
            cwd=WORKSPACE_ROOT,
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


def http_get(url: str, *, allow_hosts: Iterable[str] = DEFAULT_CONFIG.http_allow_hosts) -> str:
    if not isinstance(url, str):
        raise ToolArgumentError("url must be a string")
    host = urlparse(url).hostname
    if host not in set(allow_hosts):
        raise ToolError(f"Host is not allow-listed: {host}")
    with urlopen(url, timeout=10) as response:
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
        except Exception as exc:
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

        if call.name == "read_file":
            return read_file(args.get("path"))
        if call.name == "write_file":
            return write_file(args.get("path"), args.get("content"))
        if call.name == "run_python":
            result = run_python(
                args.get("code"),
                timeout_seconds=self.config.python_timeout_seconds,
                memory_mb=self.config.python_memory_mb,
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        if call.name == "http_get":
            return http_get(args.get("url"), allow_hosts=self.config.http_allow_hosts)
        if call.name == "send_email":
            return send_email(
                args.get("to"),
                args.get("subject"),
                args.get("body"),
                db=self.db,
                run_id=self.run_id,
                idempotency_key=idempotency_key,
            )
        raise ToolArgumentError(f"Unknown tool: {call.name}")
