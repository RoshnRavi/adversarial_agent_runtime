"""Mock LLM JSON response parsing and validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .exceptions import ToolArgumentError


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: Any = None

    def key_material(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass(frozen=True)
class ParsedAgentResponse:
    assistant_text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    partial_turn: bool = False
    expected_tool_call_count: int | None = None


def parse_agent_response(response: dict[str, Any], step: int) -> ParsedAgentResponse:
    tool_calls, parse_errors = _parse_tool_calls(response, step)
    return ParsedAgentResponse(
        assistant_text=_assistant_text(response),
        tool_calls=tool_calls,
        parse_errors=parse_errors,
        partial_turn=_is_partial_turn(response),
        expected_tool_call_count=_expected_tool_call_count(response),
    )


def _parse_tool_calls(response: dict[str, Any], step: int) -> tuple[list[ToolCall], list[str]]:
    raw_items: list[Any] = []
    errors: list[str] = []

    # mockllm deliberately varies tool-call shape across scenarios; normalize all
    # supported shapes before validating individual calls.
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
        parsed = _try_parse_json(content)
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
            args = _coerce_arguments(raw_args)
        except ToolArgumentError as exc:
            # Keep a synthetic failed tool call so malformed arguments are visible
            # to the model as tool evidence instead of disappearing.
            errors.append(str(exc))
            args = {"_malformed_arguments": str(raw_args), "_error": str(exc)}
        call_id = item.get("id")
        if not isinstance(call_id, str) or not call_id:
            # Missing IDs are made deterministic so idempotency keys stay stable
            # across replay of the same response.
            call_id = f"step-{step}-tool-{index}"
        calls.append(ToolCall(id=call_id, name=name, arguments=args, raw_arguments=raw_args))
    return calls, errors


def _coerce_arguments(raw_args: Any) -> dict[str, Any]:
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        parsed = _try_parse_json(raw_args)
        if isinstance(parsed, dict):
            return parsed
        # S02 includes common model JSON mistakes; repair only the narrow
        # trailing-comma case and report everything else as malformed.
        repaired = re.sub(r",\s*([}\]])", r"\1", raw_args)
        parsed = _try_parse_json(repaired)
        if isinstance(parsed, dict):
            return parsed
        raise ToolArgumentError("malformed tool arguments")
    raise ToolArgumentError(f"tool arguments must be an object, got {type(raw_args).__name__}")


def _assistant_text(response: dict[str, Any]) -> str:
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


def _is_partial_turn(response: dict[str, Any]) -> bool:
    finish_reason = response.get("finish_reason")
    interrupted_reasons = {
        "interrupted",
        "partial",
        "partial_tool_calls",
        "tool_call_interrupted",
        "tool_calls_interrupted",
    }
    return (
        response.get("partial") is True
        or response.get("interrupted") is True
        or finish_reason in interrupted_reasons
    )


def _expected_tool_call_count(response: dict[str, Any]) -> int | None:
    raw_count = response.get("expected_tool_call_count")
    if isinstance(raw_count, int) and not isinstance(raw_count, bool) and raw_count > 0:
        return raw_count
    expected_calls = response.get("expected_tool_calls")
    if isinstance(expected_calls, list) and expected_calls:
        return len(expected_calls)
    return None


def _try_parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
