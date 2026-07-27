"""Deterministic token counter used by the runtime memory budget.

This intentionally avoids model-specific tokenizers so tests and replay are stable across machines.
The counter treats contiguous alphanumeric runs as one token and punctuation as individual tokens.
"""

from __future__ import annotations

import re


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", re.ASCII)


def count_tokens(text: str) -> int:
    """Return a deterministic token count for `text`."""

    return len(TOKEN_PATTERN.findall(text))


def count_message_tokens(messages: list[dict[str, str]]) -> int:
    """Count tokens for Messages-API-shaped message dictionaries."""

    total = 0
    for message in messages:
        total += count_tokens(message.get("role", ""))
        total += count_tokens(message.get("content", ""))
    return total

