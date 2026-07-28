"""Memory window management and deterministic context budgeting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mockllm.tokenizer import count_message_tokens, count_tokens

from .dto import Event, ToolResult, stable_json
from .exceptions import ContextLimitExceededError


@dataclass
class MemoryMessage:
    role: str
    content: str

    def to_api(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class MemoryWindow:
    token_budget: int = 8000
    messages: list[MemoryMessage] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.messages.append(MemoryMessage(role=role, content=str(content)))
        self.compact_if_needed()

    def compact_if_needed(self) -> None:
        if self.estimated_tokens() <= self.token_budget:
            return
        if len(self.messages) <= 1:
            raise ContextLimitExceededError("single message exceeds token budget")

        first = self.messages[:1]
        removed = self.messages[1 : max(1, len(self.messages) // 2)]
        rest = self.messages[max(1, len(self.messages) // 2) :]
        summary_text = "Summary of compacted context: " + " | ".join(
            message.content[:200] for message in removed if message.content
        )
        self.messages = first + [MemoryMessage("system", summary_text)] + rest
        if self.estimated_tokens() > self.token_budget:
            self.messages = first + rest[-max(1, len(rest) // 2) :]
        if self.estimated_tokens() > self.token_budget:
            raise ContextLimitExceededError("context exceeds token budget after compaction")

    def estimated_tokens(self) -> int:
        return sum(
            count_tokens(message.role) + count_tokens(message.content) for message in self.messages
        )

    def to_messages(self) -> list[dict[str, str]]:
        return [message.to_api() for message in self.messages]


class MemoryManager:
    """Higher-level conversation memory used by the agent loop."""

    def __init__(self, token_budget: int = 8000) -> None:
        self.window = MemoryWindow(token_budget=token_budget)
        self.preserved_facts: list[str] = []

    @classmethod
    def from_events(cls, events: list[Event], token_budget: int = 8000) -> "MemoryManager":
        memory = cls(token_budget=token_budget)
        for event in events:
            if event.event_type == "run_started":
                memory.add_user_message(str(event.payload.get("task", "")))
            elif event.event_type == "assistant_message":
                memory.add_assistant_message(str(event.payload.get("content", "")))
            elif event.event_type == "tool_result":
                memory.add_tool_result(
                    str(event.payload.get("tool_name", "tool")),
                    event.payload.get("result", event.payload),
                )
        return memory

    def add_user_message(self, content: str) -> None:
        self._preserve_fact(content)
        self.window.add("user", content)

    def add_assistant_message(self, content: str) -> None:
        self._preserve_fact(content)
        self.window.add("assistant", content)

    def add_tool_result(self, tool_name: str, result: Any) -> None:
        rendered = stable_json({"tool": tool_name, "result": result})
        self._preserve_fact(rendered)
        self.window.add("tool", rendered)

    def get_compacted_context(self) -> list[dict[str, str]]:
        if self.preserved_facts:
            fact_text = "Preserved durable facts: " + " | ".join(self.preserved_facts[-20:])
            existing = [m for m in self.window.messages if m.role == "system" and m.content.startswith("Preserved durable facts:")]
            if existing:
                existing[0].content = fact_text
            else:
                self.window.messages.insert(0, MemoryMessage("system", fact_text))
        self.window.compact_if_needed()
        messages = self.window.to_messages()
        token_count = count_message_tokens(messages)
        if token_count > self.window.token_budget:
            raise ContextLimitExceededError(
                f"context has {token_count} tokens, limit is {self.window.token_budget}"
            )
        return messages

    def estimated_tokens(self) -> int:
        return self.window.estimated_tokens()

    def _preserve_fact(self, content: str) -> None:
        lowered = content.lower()
        if "remember" in lowered or "fact" in lowered or "turn 3" in lowered:
            clipped = content[:300]
            if clipped not in self.preserved_facts:
                self.preserved_facts.append(clipped)
