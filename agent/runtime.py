"""Core runtime configuration, DTOs, context bridge, and execution loop."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from mockllm.tokenizer import count_tokens

from .exceptions import (
    AgentConfigError,
    AgentError,
    BudgetExceededError,
    ContextLimitExceededError,
    MaxStepsReachedError,
    ModelContradictionError,
    NetworkFailureError,
    NoProgressError,
    PartialToolTurnError,
)
from .message import MAX_CONTEXT_TOKENS
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
    "cost_budget_tokens",
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
    "cost_budget_tokens",
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
        if value <= 0:
            raise AgentConfigError(f"{key} must be a positive integer in {source}")
        values[key] = value
    if values["token_budget"] > MAX_CONTEXT_TOKENS:
        raise AgentConfigError(
            f"token_budget must be <= {MAX_CONTEXT_TOKENS} in {source}"
        )

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
    cost_budget_tokens: int = _CONFIG_DEFAULT_VALUES["cost_budget_tokens"]
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
        status = state.status
        pending = [self._tool_call_from_dict(item) for item in state.pending_tool_calls]
        total_cost = self._cost_so_far(events)

        while step < self.config.max_steps:
            try:
                events = self.db.get_events(run_id)
                self._raise_if_recorded_no_progress(events)
                if not pending:
                    pending = self._recover_pending_tool_calls(run_id, events, step)
                    if pending:
                        pending_payload = [asdict(call) for call in pending]
                        self.db.update_run(
                            run_id,
                            status="RUNNING_TOOLS",
                            step_count=step,
                            pending_tool_calls=pending_payload,
                        )
                        status = "RUNNING_TOOLS"

                if pending:
                    self._execute_tool_calls(run_id, pending, memory, step)
                    pending = []
                    self.db.update_run(
                        run_id,
                        status="CALLING_LLM",
                        step_count=step,
                        pending_tool_calls=[],
                    )
                    status = "CALLING_LLM"
                    continue

                events = self.db.get_events(run_id)
                recorded_response = self._recorded_response_to_process(events, step)
                if recorded_response is not None:
                    request_payload = self._llm_request_payload(events, step)
                    validated_response = self.validator.validate_recorded_response(
                        run_id,
                        step=step,
                        memory=memory,
                        response=recorded_response,
                        message_count=int(request_payload.get("message_count", 0)),
                        token_count=int(request_payload.get("tokens", 0)),
                    )
                else:
                    if status == "CALLING_LLM" and step > 0 and not self._has_llm_response(
                        events,
                        step,
                    ):
                        next_step = step
                    else:
                        next_step = step + 1
                    step = next_step
                    self.db.update_run(run_id, status="CALLING_LLM", step_count=step)
                    status = "CALLING_LLM"
                    self._log_loop_control(
                        run_id,
                        step,
                        {
                            "decision": "step_start",
                            "step": step,
                            "max_steps": self.config.max_steps,
                            "total_cost": total_cost,
                            "cost_budget": self.config.cost_budget_tokens,
                        },
                    )
                    validated_response = self.validator.validate_turn(
                        run_id,
                        task=task,
                        step=step,
                        memory=memory,
                    )
                total_cost = self._charge_budget(run_id, validated_response, total_cost, step)

                if self._is_incomplete_partial_turn(validated_response):
                    payload = {
                        "expected_tool_call_count": validated_response.expected_tool_call_count,
                        "parsed_tool_call_count": len(validated_response.tool_calls),
                    }
                    inserted = self.db.append_event(
                        run_id,
                        "partial_tool_turn",
                        payload,
                        step=step,
                        idempotency_key=f"{run_id}:partial_tool_turn:{step}",
                    )
                    if inserted is not None:
                        self.tracer.log_trace(run_id, "partial_tool_turn", payload, step=step)
                    raise PartialToolTurnError("incomplete interrupted tool-call turn")

                if not validated_response.tool_calls:
                    contradiction = self._model_contradiction(
                        run_id,
                        validated_response.assistant_text,
                        step,
                    )
                    if contradiction is not None:
                        inserted = self.db.append_event(
                            run_id,
                            "model_contradiction",
                            contradiction,
                            step=step,
                            idempotency_key=self._event_key(
                                run_id,
                                "model_contradiction",
                                step,
                                contradiction,
                            ),
                        )
                        if inserted is not None:
                            self.tracer.log_trace(
                                run_id,
                                "model_contradiction",
                                contradiction,
                                step=step,
                            )
                        raise ModelContradictionError(
                            "assistant claimed success despite failed tool result"
                        )
                    reason = "completed"
                    self.db.finish_run(run_id, reason, step_count=step)
                    self.tracer.log_trace(run_id, "run_finished", {"reason": reason}, step=step)
                    return self.db.get_run(run_id) or state

                pending_payload = [asdict(call) for call in validated_response.tool_calls]
                parsed_inserted = self.db.update_run_and_append_event(
                    run_id,
                    "tool_calls_parsed",
                    {"tool_calls": pending_payload},
                    status="RUNNING_TOOLS",
                    step_count=step,
                    pending_tool_calls=pending_payload,
                    step=step,
                    idempotency_key=f"{run_id}:tool_calls_parsed:{step}",
                )
                if parsed_inserted is not None:
                    self.tracer.log_trace(
                        run_id,
                        "tool_calls_parsed",
                        {"tool_calls": pending_payload},
                        step=step,
                    )
                self._check_progress(run_id, validated_response.tool_calls, step)
                status = "RUNNING_TOOLS"
                pending = validated_response.tool_calls
            except (AgentError, NetworkFailureError, ContextLimitExceededError) as exc:
                reason = f"{type(exc).__name__}: {exc}"
                self.db.finish_run(run_id, reason, step_count=step)
                self.tracer.log_trace(run_id, "run_finished", {"reason": reason}, step=step)
                return self.db.get_run(run_id) or state

        reason = f"{MaxStepsReachedError.__name__}: step limit reached"
        self._log_loop_control(
            run_id,
            step,
            {
                "decision": "step_limit",
                "step": step,
                "max_steps": self.config.max_steps,
            },
        )
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
        from .run import AgentDatabase

        for call in calls:
            idempotency_key = self._idempotency_key(run_id, call)
            start_payload = {"tool_call": asdict(call)}
            start_inserted = self.db.append_event(
                run_id,
                "tool_execution_started",
                start_payload,
                step=step,
                idempotency_key=f"start:{idempotency_key}",
            )
            if start_inserted is not None:
                self.tracer.log_trace(run_id, "tool_execution_started", start_payload, step=step)

        def run_one(index: int, call: ToolCall) -> tuple[int, ToolResult]:
            worker_db = AgentDatabase(self.config.db_path)
            try:
                executor = ToolExecutor(worker_db, run_id=run_id, config=self.config)
                idempotency_key = self._idempotency_key(run_id, call)
                existing = worker_db.get_tool_execution(idempotency_key)
                if existing is not None:
                    result = executor.execute(call, idempotency_key)
                elif call.name == "send_email" and memory.has_untrusted_tool_results():
                    result = self._blocked_tool_result(
                        run_id,
                        call,
                        idempotency_key,
                        "SecurityViolationError: send_email blocked after untrusted tool results",
                        db=worker_db,
                    )
                else:
                    result = executor.execute(call, idempotency_key)
                return index, result
            finally:
                worker_db.close()

        ordered_results: list[ToolResult | None] = [None] * len(calls)
        with ThreadPoolExecutor(max_workers=len(calls)) as pool:
            futures = [pool.submit(run_one, index, call) for index, call in enumerate(calls)]
            for future in futures:
                index, result = future.result()
                ordered_results[index] = result

        results = [result for result in ordered_results if result is not None]
        for result in results:
            idempotency_key = result.idempotency_key or self._event_key(
                run_id,
                "tool_result",
                step,
                {
                    "tool_call_id": result.tool_call_id,
                    "tool_name": result.tool_name,
                },
            )
            payload = {
                "tool_call_id": result.tool_call_id,
                "tool_name": result.tool_name,
                "ok": result.ok,
                "result": result.result,
                "error": result.error,
                "idempotency_key": result.idempotency_key,
            }
            result_inserted = self.db.append_event(
                run_id,
                "tool_result",
                payload,
                step=step,
                idempotency_key=f"result:{idempotency_key}",
            )
            if result_inserted is not None:
                self.tracer.log_trace(run_id, "tool_result", payload, step=step)
                memory.add_tool_result(result.tool_name, payload)
        return results

    def _blocked_tool_result(
        self,
        run_id: str,
        call: ToolCall,
        idempotency_key: str,
        error: str,
        *,
        db: Any | None = None,
    ) -> ToolResult:
        database = db or self.db
        existing = database.get_tool_execution(idempotency_key)
        if existing is not None:
            return ToolResult(
                tool_call_id=call.id,
                tool_name=call.name,
                ok=existing["status"] == "completed" and not existing["error"],
                result=existing["result"],
                error=existing["error"],
                idempotency_key=idempotency_key,
            )
        database.record_tool_execution(
            idempotency_key=idempotency_key,
            run_id=run_id,
            tool_call_id=call.id,
            tool_name=call.name,
            arguments=call.arguments,
            error=error,
            status="blocked",
        )
        return ToolResult(
            tool_call_id=call.id,
            tool_name=call.name,
            ok=False,
            error=error,
            idempotency_key=idempotency_key,
        )

    def _check_progress(self, run_id: str, calls: list[ToolCall], step: int) -> None:
        digest = self._tool_progress_digest(calls)
        repeat_count = 1
        previous = self._latest_progress_check(self.db.get_events(run_id), before_step=step)
        if previous is not None and previous.get("tool_digest") == digest:
            previous_count = self._positive_int(previous.get("repeat_count"), 0)
            repeat_count = previous_count + 1 if previous_count else 1
        self._log_loop_control(
            run_id,
            step,
            {
                "decision": "progress_check",
                "tool_digest": digest,
                "tool_count": len(calls),
                "repeat_count": repeat_count,
                "no_progress_limit": self.config.no_progress_limit,
            },
        )
        if repeat_count >= self.config.no_progress_limit:
            raise NoProgressError("same tool call repeated without progress")

    def _raise_if_recorded_no_progress(self, events: list[Event]) -> None:
        progress = self._latest_progress_check(events)
        if progress is None:
            return
        repeat_count = self._positive_int(progress.get("repeat_count"), 0)
        limit = self._positive_int(
            progress.get("no_progress_limit"),
            self.config.no_progress_limit,
        )
        if repeat_count >= limit:
            raise NoProgressError("same tool call repeated without progress")

    @staticmethod
    def _tool_progress_digest(calls: list[ToolCall]) -> str:
        material = [{"name": call.name, "arguments": call.arguments} for call in calls]
        return hashlib.sha256(stable_json(material).encode("utf-8")).hexdigest()

    @staticmethod
    def _latest_progress_check(
        events: list[Event],
        *,
        before_step: int | None = None,
    ) -> dict[str, Any] | None:
        for event in reversed(events):
            if event.event_type != "loop_control":
                continue
            if event.payload.get("decision") != "progress_check":
                continue
            if before_step is not None and (
                event.step is None or event.step >= before_step
            ):
                continue
            return event.payload
        return None

    @staticmethod
    def _positive_int(value: Any, default: int) -> int:
        if type(value) is int and value > 0:
            return value
        return default

    def _is_incomplete_partial_turn(self, response: Any) -> bool:
        if not getattr(response, "partial_turn", False):
            return False
        expected_count = getattr(response, "expected_tool_call_count", None)
        actual_count = len(getattr(response, "tool_calls", []))
        return expected_count is None or actual_count < expected_count

    def _model_contradiction(
        self,
        run_id: str,
        assistant_text: str,
        step: int,
    ) -> dict[str, Any] | None:
        text = assistant_text.lower()
        success_markers = ("successfully", "succeeded", "success", "completed", "worked")
        if not assistant_text or not any(marker in text for marker in success_markers):
            return None

        failed_results = self._latest_failed_tool_results(run_id, step)
        for failed in failed_results:
            targets = self._failed_tool_targets(failed)
            matched = [target for target in targets if target and target.lower() in text]
            if matched:
                return {
                    "reason": "assistant claimed success for a failed tool result",
                    "assistant_text": assistant_text,
                    "matched_targets": matched,
                    "failed_tool_result": failed.get("tool_result", {}),
                    "failed_tool_arguments": failed.get("arguments", {}),
                }
        return None

    def _latest_failed_tool_results(self, run_id: str, current_step: int) -> list[dict[str, Any]]:
        events = self.db.get_events(run_id)
        latest_tool_step = max(
            (
                int(event.step)
                for event in events
                if event.event_type == "tool_result"
                and event.step is not None
                and event.step < current_step
            ),
            default=None,
        )
        if latest_tool_step is None:
            return []

        starts: dict[str, dict[str, Any]] = {}
        for event in events:
            if event.event_type != "tool_execution_started" or event.step != latest_tool_step:
                continue
            tool_call = event.payload.get("tool_call")
            if isinstance(tool_call, dict):
                starts[str(tool_call.get("id"))] = tool_call

        failed: list[dict[str, Any]] = []
        for event in events:
            if event.event_type != "tool_result" or event.step != latest_tool_step:
                continue
            if event.payload.get("ok") is not False:
                continue
            tool_call_id = str(event.payload.get("tool_call_id", ""))
            start = starts.get(tool_call_id, {})
            arguments = start.get("arguments") if isinstance(start, dict) else {}
            failed.append(
                {
                    "tool_result": event.payload,
                    "arguments": arguments if isinstance(arguments, dict) else {},
                }
            )
        return failed

    @staticmethod
    def _failed_tool_targets(failed: dict[str, Any]) -> list[str]:
        result = failed.get("tool_result", {})
        arguments = failed.get("arguments", {})
        if not isinstance(result, dict) or not isinstance(arguments, dict):
            return []

        targets: list[str] = []
        tool_name = str(result.get("tool_name", ""))
        if tool_name in {"read_file", "write_file"} and isinstance(arguments.get("path"), str):
            targets.append(arguments["path"])
        if tool_name == "http_get" and isinstance(arguments.get("url"), str):
            targets.append(arguments["url"])
        if tool_name == "send_email":
            targets.extend(
                str(arguments[key])
                for key in ("to", "subject", "body")
                if isinstance(arguments.get(key), str)
            )
        if tool_name == "run_python" and isinstance(arguments.get("code"), str):
            targets.append(arguments["code"][:80])
        error = result.get("error")
        if isinstance(error, str):
            targets.extend(part for part in _interesting_error_parts(error) if part)
        return list(dict.fromkeys(targets))

    def _charge_budget(self, run_id: str, response: Any, total_cost: int, step: int) -> int:
        existing = self._event_payload(self.db.get_events(run_id), "budget_charged", step)
        if existing is not None:
            existing_total = int(existing.get("total_cost", total_cost))
            if existing_total > self.config.cost_budget_tokens:
                raise BudgetExceededError("cost budget exceeded")
            return existing_total
        request_cost = int(response.token_count)
        response_cost = count_tokens(stable_json(response.raw_response))
        step_cost = request_cost + response_cost
        next_total = total_cost + step_cost
        payload = {
            "request_cost": request_cost,
            "response_cost": response_cost,
            "step_cost": step_cost,
            "total_cost": next_total,
            "budget": self.config.cost_budget_tokens,
        }
        inserted = self.db.append_event(
            run_id,
            "budget_charged",
            payload,
            step=step,
            idempotency_key=f"{run_id}:budget_charged:{step}",
        )
        if inserted is not None:
            self.tracer.log_trace(run_id, "budget_charged", payload, step=step)
        if next_total > self.config.cost_budget_tokens:
            raise BudgetExceededError("cost budget exceeded")
        return next_total

    def _cost_so_far(self, events: list[Any]) -> int:
        total = 0
        for event in events:
            if event.event_type == "budget_charged":
                total = int(event.payload.get("total_cost", total))
        return total

    def _log_loop_control(self, run_id: str, step: int, payload: dict[str, Any]) -> None:
        inserted = self.db.append_event(
            run_id,
            "loop_control",
            payload,
            step=step,
            idempotency_key=self._event_key(run_id, "loop_control", step, payload),
        )
        if inserted is not None:
            self.tracer.log_trace(run_id, "loop_control", payload, step=step)

    def _idempotency_key(self, run_id: str, call: ToolCall) -> str:
        if call.name == "send_email":
            material = {
                "run_id": run_id,
                "tool_name": call.name,
                "logical_email": {
                    "to": call.arguments.get("to"),
                    "subject": call.arguments.get("subject"),
                    "body": call.arguments.get("body"),
                },
            }
        else:
            material = {"run_id": run_id, "tool_call": call.key_material()}
        return hashlib.sha256(stable_json(material).encode("utf-8")).hexdigest()

    def _event_key(
        self,
        run_id: str,
        event_type: str,
        step: int,
        payload: dict[str, Any],
    ) -> str:
        digest = hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()
        return f"{run_id}:{event_type}:{step}:{digest}"

    def _recover_pending_tool_calls(
        self,
        run_id: str,
        events: list[Event],
        step: int,
    ) -> list[ToolCall]:
        parsed_payload = self._event_payload(events, "tool_calls_parsed", step)
        if parsed_payload is None:
            return []
        raw_calls = parsed_payload.get("tool_calls")
        if not isinstance(raw_calls, list):
            return []
        calls = [
            self._tool_call_from_dict(item)
            for item in raw_calls
            if isinstance(item, dict)
        ]
        completed_keys = {
            str(event.payload.get("idempotency_key"))
            for event in events
            if event.event_type == "tool_result"
            and event.step == step
            and event.payload.get("idempotency_key") is not None
        }
        completed_call_ids = {
            str(event.payload.get("tool_call_id"))
            for event in events
            if event.event_type == "tool_result"
            and event.step == step
            and event.payload.get("idempotency_key") is None
            and event.payload.get("tool_call_id") is not None
        }
        return [
            call
            for call in calls
            if self._idempotency_key(run_id, call) not in completed_keys
            and call.id not in completed_call_ids
        ]

    def _recorded_response_to_process(
        self,
        events: list[Event],
        step: int,
    ) -> dict[str, Any] | None:
        if step <= 0:
            return None
        response_payload = self._event_payload(events, "llm_response", step)
        if response_payload is None:
            return None
        if self._has_event(events, "tool_calls_parsed", step):
            return None
        if self._has_event(events, "run_finished", step):
            return None
        return response_payload

    def _llm_request_payload(self, events: list[Event], step: int) -> dict[str, Any]:
        return self._event_payload(events, "llm_request", step) or {
            "message_count": 0,
            "tokens": 0,
        }

    def _has_llm_response(self, events: list[Event], step: int) -> bool:
        return self._has_event(events, "llm_response", step)

    def _has_event(self, events: list[Event], event_type: str, step: int) -> bool:
        return self._event_payload(events, event_type, step) is not None

    @staticmethod
    def _event_payload(
        events: list[Event],
        event_type: str,
        step: int,
    ) -> dict[str, Any] | None:
        for event in reversed(events):
            if event.event_type == event_type and event.step == step:
                return event.payload
        return None

    def _tool_call_from_dict(self, item: dict[str, Any]) -> ToolCall:
        return ToolCall(
            id=str(item.get("id")),
            name=str(item.get("name")),
            arguments=dict(item.get("arguments") or {}),
            raw_arguments=item.get("raw_arguments"),
        )


def _interesting_error_parts(error: str) -> list[str]:
    parts: list[str] = []
    for raw_token in error.replace('"', " ").replace("'", " ").split():
        token = raw_token.strip(" ,;:()[]{}")
        if not token:
            continue
        if "/" in token:
            name = Path(token).name
            if name:
                parts.append(name)
        if "." in token and len(token) > 2:
            parts.append(token)
    return parts
