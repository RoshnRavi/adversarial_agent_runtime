"""JSONL structured trace writing and replay support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .config import DEFAULT_CONFIG
from .dto import TraceEvent, stable_json


TRACE_DIR = DEFAULT_CONFIG.trace_dir


class TraceWriter:
    def __init__(self, directory: Path = TRACE_DIR) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def write(self, run_id: str, event: dict[str, Any]) -> None:
        path = self.directory / f"{run_id}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(stable_json(event) + "\n")

    def log_trace(
        self,
        run_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        step: int | None = None,
    ) -> None:
        self.write(
            run_id,
            TraceEvent(
                run_id=run_id,
                event_type=event_type,
                payload=payload or {},
                step=step,
            ).to_dict(),
        )


class TraceReader:
    def __init__(self, directory: Path = TRACE_DIR) -> None:
        self.directory = Path(directory)

    def read(self, run_id: str) -> Iterator[dict[str, Any]]:
        path = self.directory / f"{run_id}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
