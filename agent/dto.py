"""Dataclasses shared by the runtime, persistence layer, and traces."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any


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
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: Any = None

    def key_material(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


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
