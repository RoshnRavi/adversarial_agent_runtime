"""Finite-state agent loop for Part A."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from typing import Any

from .config import DEFAULT_CONFIG, RuntimeConfig
from .database import AgentDatabase
from .dto import RunState, ToolCall, ToolResult, stable_json
from .exceptions import (
    AgentError,
    ContextLimitExceededError,
    MaxStepsReachedError,
    NetworkFailureError,
    NoProgressError,
    ToolArgumentError,
)
from .llm_client import LLMClient
from .memory import MemoryManager
from .tools import ToolExecutor
from .trace import TraceWriter


class AgentLoop:
    """Coordinates model calls, tool execution, durability, and traces."""

    def __init__(
        self,
        db: AgentDatabase,
        *,
        memory: MemoryManager | None = None,
        tracer: TraceWriter | None = None,
        llm_client: LLMClient | None = None,
        config: RuntimeConfig = DEFAULT_CONFIG,
    ) -> None:
        self.db = db
        self.memory = memory
        self.tracer = tracer or TraceWriter(config.trace_dir)
        self.llm_client = llm_client or LLMClient(config)
        self.config = config
        self._last_tool_hash: str | None = None
        self._repeat_count = 0

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
        memory = self.memory or MemoryManager.from_events(events, self.config.token_budget)
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
                context = memory.get_compacted_context()
                self.db.append_event(
                    run_id,
                    "llm_request",
                    {"message_count": len(context), "tokens": memory.estimated_tokens()},
                    step=step,
                )
                self.tracer.log_trace(
                    run_id,
                    "llm_request",
                    {"message_count": len(context), "tokens": memory.estimated_tokens()},
                    step=step,
                )

                response = self.llm_client.call({"messages": context, "task": task, "step": step})
                for retry in getattr(self.llm_client, "last_retries", []):
                    self.tracer.log_trace(run_id, "retry", asdict(retry), step=step)
                self.db.append_event(run_id, "llm_response", response, step=step)
                self.tracer.log_trace(run_id, "llm_response", response, step=step)

                assistant_text = self._assistant_text(response)
                if assistant_text:
                    memory.add_assistant_message(assistant_text)
                    self.db.append_event(
                        run_id,
                        "assistant_message",
                        {"content": assistant_text},
                        step=step,
                    )

                tool_calls, parse_errors = self._parse_tool_calls(response, step)
                for error in parse_errors:
                    self.db.append_event(run_id, "tool_parse_error", {"error": error}, step=step)
                    self.tracer.log_trace(run_id, "tool_parse_error", {"error": error}, step=step)

                if not tool_calls:
                    reason = "completed"
                    self.db.finish_run(run_id, reason, step_count=step)
                    self.tracer.log_trace(run_id, "run_finished", {"reason": reason}, step=step)
                    return self.db.get_run(run_id) or state

                self._check_progress(tool_calls)
                pending_payload = [asdict(call) for call in tool_calls]
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
                pending = tool_calls
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

    def _parse_tool_calls(self, response: dict[str, Any], step: int) -> tuple[list[ToolCall], list[str]]:
        raw_items: list[Any] = []
        errors: list[str] = []

        if isinstance(response.get("tool_calls"), list):
            raw_items.extend(response["tool_calls"])
        if isinstance(response.get("tool_call"), dict):
            raw_items.append(response["tool_call"])
        if isinstance(response.get("tool_use"), list):
            raw_items.extend(response["tool_use"])

        content = response.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") in {"tool_use", "tool_call"}:
                    raw_items.append(item)
        elif isinstance(content, str):
            parsed = self._try_parse_json(content)
            if isinstance(parsed, dict):
                if isinstance(parsed.get("tool_call"), dict):
                    raw_items.append(parsed["tool_call"])
                if isinstance(parsed.get("tool_calls"), list):
                    raw_items.extend(parsed["tool_calls"])

        calls: list[ToolCall] = []
        for index, item in enumerate(raw_items):
            if not isinstance(item, dict):
                errors.append(f"tool call {index} is not an object")
                continue
            name = item.get("name") or item.get("tool_name")
            if not isinstance(name, str) or not name:
                errors.append(f"tool call {index} is missing a valid name")
                continue
            raw_args = item.get("arguments", item.get("input", item.get("args", {})))
            try:
                args = self._coerce_arguments(raw_args)
            except ToolArgumentError as exc:
                errors.append(str(exc))
                args = {"_malformed_arguments": str(raw_args), "_error": str(exc)}
            call_id = item.get("id")
            if not isinstance(call_id, str) or not call_id:
                call_id = f"step-{step}-tool-{index}"
            calls.append(ToolCall(id=call_id, name=name, arguments=args, raw_arguments=raw_args))
        return calls, errors

    def _coerce_arguments(self, raw_args: Any) -> dict[str, Any]:
        if raw_args is None:
            return {}
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str):
            parsed = self._try_parse_json(raw_args)
            if isinstance(parsed, dict):
                return parsed
            repaired = re.sub(r",\s*([}\]])", r"\1", raw_args)
            parsed = self._try_parse_json(repaired)
            if isinstance(parsed, dict):
                return parsed
            raise ToolArgumentError("malformed tool arguments")
        raise ToolArgumentError(f"tool arguments must be an object, got {type(raw_args).__name__}")

    def _assistant_text(self, response: dict[str, Any]) -> str:
        content = response.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            return "\n".join(part for part in parts if part)
        return str(response.get("message", ""))

    def _try_parse_json(self, text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

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
