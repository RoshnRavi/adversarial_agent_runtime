"""Replay recorded runs without contacting the model server."""

from __future__ import annotations

from .exceptions import ReplayError
from .trace import TraceReader


class Replayer:
    def __init__(self, reader: TraceReader | None = None) -> None:
        self.reader = reader or TraceReader()

    def replay(self, run_id: str) -> list[str]:
        try:
            events = list(self.reader.read(run_id))
        except FileNotFoundError as exc:
            raise ReplayError(f"No trace found for run_id {run_id}") from exc

        lines: list[str] = []
        for event in events:
            step = event.get("step")
            prefix = f"step {step}: " if step is not None else ""
            lines.append(f"{prefix}{event.get('event_type')} {event.get('payload', {})}")
        return lines


def replay_run(run_id: str) -> str:
    return "\n".join(Replayer().replay(run_id))
