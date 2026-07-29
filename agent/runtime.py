"""Core runtime configuration, DTOs, context bridge, and execution loop."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from .exceptions import (
    AgentConfigError,
    AgentError,
    ContextLimitExceededError,
    MaxStepsReachedError,
    NetworkFailureError,
    NoProgressError,
)
from .message import MemoryManager as _MemoryManager
from .response import ToolCall

if TYPE_CHECKING:
    from .message import MemoryManager
    from .run import AgentDatabase, TraceWriter
    from .validate import AgentValidator, LLMClient


CONFIG_PATH = Path(__file__).with_name("config_agent.yaml")
_CONFIG_KEYS = {
    "db_path",
    "trace_dir",
    "workspace_root",
    "server_url",
    "token_budget",
    "max_steps",
    "max_retries",
    "retry_base_delay",
    "request_timeout_seconds",
    "circuit_max_failures",
    "circuit_cooldown_seconds",
    "no_progress_limit",
    "python_timeout_seconds",
    "python_memory_mb",
    "http_timeout_seconds",
    "http_allow_hosts",
}
_PATH_KEYS = {"db_path", "trace_dir", "workspace_root"}
_INT_KEYS = {
    "token_budget",
    "max_steps",
    "max_retries",
    "circuit_max_failures",
    "circuit_cooldown_seconds",
    "no_progress_limit",
    "python_timeout_seconds",
    "python_memory_mb",
}
_FLOAT_KEYS = {"retry_base_delay", "request_timeout_seconds", "http_timeout_seconds"}


def _read_config_mapping(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AgentConfigError(f"runtime config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise AgentConfigError(f"runtime config file is invalid YAML: {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise AgentConfigError(f"runtime config file must contain a mapping: {path}")
    return loaded


def _coerce_runtime_config_values(raw: dict[str, Any], *, source: Path | str) -> dict[str, Any]:
    missing = sorted(_CONFIG_KEYS - set(raw))
    if missing:
        raise AgentConfigError(f"missing runtime config key(s) in {source}: {', '.join(missing)}")
    unknown = sorted(set(raw) - _CONFIG_KEYS)
    if unknown:
        raise AgentConfigError(f"unknown runtime config key(s) in {source}: {', '.join(unknown)}")

    values: dict[str, Any] = {}
    for key in _PATH_KEYS:
        value = raw[key]
        if not isinstance(value, (str, Path)):
            raise AgentConfigError(f"{key} must be a path string in {source}")
        values[key] = Path(value)

    server_url = raw["server_url"]
    if not isinstance(server_url, str) or not server_url:
        raise AgentConfigError(f"server_url must be a non-empty string in {source}")
    values["server_url"] = server_url

    for key in _INT_KEYS:
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise AgentConfigError(f"{key} must be an integer in {source}")
        values[key] = value

    for key in _FLOAT_KEYS:
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AgentConfigError(f"{key} must be a number in {source}")
        values[key] = float(value)

    allow_hosts = raw["http_allow_hosts"]
    if not isinstance(allow_hosts, (list, tuple)) or not allow_hosts:
        raise AgentConfigError(f"http_allow_hosts must be a non-empty list in {source}")
    if not all(isinstance(host, str) and host for host in allow_hosts):
        raise AgentConfigError(f"http_allow_hosts entries must be non-empty strings in {source}")
    values["http_allow_hosts"] = tuple(allow_hosts)
    return values


_CONFIG_DEFAULT_VALUES = _coerce_runtime_config_values(
    _read_config_mapping(CONFIG_PATH),
    source=CONFIG_PATH,
)


@dataclass(frozen=True)
class RuntimeConfig:
    db_path: Path = _CONFIG_DEFAULT_VALUES["db_path"]
    trace_dir: Path = _CONFIG_DEFAULT_VALUES["trace_dir"]
    workspace_root: Path = _CONFIG_DEFAULT_VALUES["workspace_root"]
    server_url: str = _CONFIG_DEFAULT_VALUES["server_url"]
    token_budget: int = _CONFIG_DEFAULT_VALUES["token_budget"]
    max_steps: int = _CONFIG_DEFAULT_VALUES["max_steps"]
    max_retries: int = _CONFIG_DEFAULT_VALUES["max_retries"]
    retry_base_delay: float = _CONFIG_DEFAULT_VALUES["retry_base_delay"]
    request_timeout_seconds: float = _CONFIG_DEFAULT_VALUES["request_timeout_seconds"]
    circuit_max_failures: int = _CONFIG_DEFAULT_VALUES["circuit_max_failures"]
    circuit_cooldown_seconds: int = _CONFIG_DEFAULT_VALUES["circuit_cooldown_seconds"]
    no_progress_limit: int = _CONFIG_DEFAULT_VALUES["no_progress_limit"]
    python_timeout_seconds: int = _CONFIG_DEFAULT_VALUES["python_timeout_seconds"]
    python_memory_mb: int = _CONFIG_DEFAULT_VALUES["python_memory_mb"]
    http_timeout_seconds: float = _CONFIG_DEFAULT_VALUES["http_timeout_seconds"]
    http_allow_hosts: tuple[str, ...] = _CONFIG_DEFAULT_VALUES["http_allow_hosts"]

    def __post_init__(self) -> None:
        values = _coerce_runtime_config_values(
            {key: getattr(self, key) for key in _CONFIG_KEYS},
            source="RuntimeConfig",
        )
        for key, value in values.items():
            object.__setattr__(self, key, value)


def load_runtime_config(path: Path | None = None, **overrides: Any) -> RuntimeConfig:
    config_path = path or CONFIG_PATH
    raw = _read_config_mapping(config_path)
    clean_overrides = {key: value for key, value in overrides.items() if value is not None}
    unknown_overrides = sorted(set(clean_overrides) - _CONFIG_KEYS)
    if unknown_overrides:
        raise AgentConfigError(
            f"unknown runtime config override(s): {', '.join(unknown_overrides)}"
        )
    values = _coerce_runtime_config_values(
        {**raw, **clean_overrides},
        source=config_path,
    )
    return RuntimeConfig(**values)


DEFAULT_CONFIG = load_runtime_config()


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class RunState:
    run_id: str
    task: str
    status: str
    step_count: int = 0
    termination_reason: str | None = None
    pending_tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class Event:
    run_id: str
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    step: int | None = None
    idempotency_key: str | None = None
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    tool_name: str
    ok: bool
    result: Any = None
    error: str | None = None
    idempotency_key: str | None = None


@dataclass(frozen=True)
class TraceEvent:
    run_id: str
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    step: int | None = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentRuntime:
    """Coordinates model calls, tool execution, durability, context, and traces."""

    def __init__(
        self,
        db: AgentDatabase,
        *,
        memory: MemoryManager | None = None,
        tracer: TraceWriter | None = None,
        llm_client: LLMClient | None = None,
        config: RuntimeConfig = DEFAULT_CONFIG,
    ) -> None:
        if tracer is None:
            from .run import TraceWriter

            tracer = TraceWriter(config.trace_dir)
        if llm_client is None:
            from .validate import LLMClient

            llm_client = LLMClient(config)
        from .validate import AgentValidator

        self.db = db
        self.memory = memory
        self.tracer = tracer
        self.llm_client = llm_client
        self.validator: AgentValidator = AgentValidator(db, tracer, llm_client)
        self.config = config
        self._last_tool_hash: str | None = None
        self._repeat_count = 0

    def start_task(self, task: str) -> RunState:
        return self.run_task(str(uuid.uuid4()), task)

    def run_task(self, run_id: str, task: str) -> RunState:
        return self.run(run_id, task=task)

    def resume(self, run_id: str) -> RunState:
        state = self.db.get_run(run_id)
        if state is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        return self.run(run_id, task=state.task)

    def run(self, run_id: str, *, task: str) -> RunState:
        created = self.db.create_run(run_id, task)
        state = self.db.get_run(run_id)
        if state is None:
            raise KeyError(f"Unable to create run: {run_id}")
        if state.status == "FINISHED":
            return state

        events = self.db.get_events(run_id)
        if self.memory is not None:
            memory = self.memory
        elif created:
            memory = _MemoryManager(self.config.token_budget)
        else:
            memory = _MemoryManager.from_events(events, self.config.token_budget)
        if created:
            memory.add_user_message(task)
            self.tracer.log_trace(run_id, "run_started", {"task": task}, step=0)

        step = state.step_count
        pending = [self._tool_call_from_dict(item) for item in state.pending_tool_calls]

        while step < self.config.max_steps:
            try:
                if pending:
                    self._execute_tool_calls(run_id, pending, memory, step)
                    pending = []
                    self.db.update_run(
                        run_id,
                        status="CALLING_LLM",
                        step_count=step,
                        pending_tool_calls=[],
                    )
                    continue

                step += 1
                self.db.update_run(run_id, status="CALLING_LLM", step_count=step)
                validated_response = self.validator.validate_turn(
                    run_id,
                    task=task,
                    step=step,
                    memory=memory,
                )

                if not validated_response.tool_calls:
                    reason = "completed"
                    self.db.finish_run(run_id, reason, step_count=step)
                    self.tracer.log_trace(run_id, "run_finished", {"reason": reason}, step=step)
                    return self.db.get_run(run_id) or state

                self._check_progress(validated_response.tool_calls)
                pending_payload = [asdict(call) for call in validated_response.tool_calls]
                self.db.update_run(
                    run_id,
                    status="RUNNING_TOOLS",
                    step_count=step,
                    pending_tool_calls=pending_payload,
                )
                self.db.append_event(
                    run_id,
                    "tool_calls_parsed",
                    {"tool_calls": pending_payload},
                    step=step,
                )
                self.tracer.log_trace(
                    run_id,
                    "tool_calls_parsed",
                    {"tool_calls": pending_payload},
                    step=step,
                )
                pending = validated_response.tool_calls
            except (AgentError, NetworkFailureError, ContextLimitExceededError) as exc:
                reason = f"{type(exc).__name__}: {exc}"
                self.db.finish_run(run_id, reason, step_count=step)
                self.tracer.log_trace(run_id, "run_finished", {"reason": reason}, step=step)
                return self.db.get_run(run_id) or state

        reason = f"{MaxStepsReachedError.__name__}: step limit reached"
        self.db.finish_run(run_id, reason, step_count=step)
        self.tracer.log_trace(run_id, "run_finished", {"reason": reason}, step=step)
        return self.db.get_run(run_id) or state

    def _execute_tool_calls(
        self,
        run_id: str,
        calls: list[ToolCall],
        memory: MemoryManager,
        step: int,
    ) -> list[ToolResult]:
        from .executer import ToolExecutor

        executor = ToolExecutor(self.db, run_id=run_id, config=self.config)
        results: list[ToolResult] = []
        for call in calls:
            idempotency_key = self._idempotency_key(run_id, call)
            self.db.append_event(
                run_id,
                "tool_execution_started",
                {"tool_call": asdict(call)},
                step=step,
                idempotency_key=f"start:{idempotency_key}",
            )
            result = executor.execute(call, idempotency_key)
            results.append(result)
            payload = {
                "tool_call_id": result.tool_call_id,
                "tool_name": result.tool_name,
                "ok": result.ok,
                "result": result.result,
                "error": result.error,
                "idempotency_key": result.idempotency_key,
            }
            self.db.append_event(run_id, "tool_result", payload, step=step)
            self.tracer.log_trace(run_id, "tool_result", payload, step=step)
            memory.add_tool_result(call.name, payload)
        return results

    def _check_progress(self, calls: list[ToolCall]) -> None:
        digest = hashlib.sha256(
            stable_json(
                [{"name": call.name, "arguments": call.arguments} for call in calls]
            ).encode("utf-8")
        ).hexdigest()
        if digest == self._last_tool_hash:
            self._repeat_count += 1
        else:
            self._last_tool_hash = digest
            self._repeat_count = 1
        if self._repeat_count >= self.config.no_progress_limit:
            raise NoProgressError("same tool call repeated without progress")

    def _idempotency_key(self, run_id: str, call: ToolCall) -> str:
        material = {"run_id": run_id, "tool_call": call.key_material()}
        return hashlib.sha256(stable_json(material).encode("utf-8")).hexdigest()

    def _tool_call_from_dict(self, item: dict[str, Any]) -> ToolCall:
        return ToolCall(
            id=str(item.get("id")),
            name=str(item.get("name")),
            arguments=dict(item.get("arguments") or {}),
            raw_arguments=item.get("raw_arguments"),
        )
