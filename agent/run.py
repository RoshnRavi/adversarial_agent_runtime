"""Run persistence, JSONL traces, and replay support."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exceptions import ReplayError
from .runtime import DEFAULT_CONFIG, Event, RunState, TraceEvent, stable_json

TRACE_DIR = DEFAULT_CONFIG.trace_dir


@dataclass(frozen=True)
class RunStateDTO:
    """Compatibility DTO for older tests and entrypoints."""

    run_id: str
    current_state: str
    step_count: int


@dataclass(frozen=True)
class EventDTO:
    """Compatibility DTO for append-only events."""

    run_id: str
    event_type: str
    payload: str | dict[str, Any] = ""
    idempotency_key: str | None = None


class AgentDatabase:
    """Manages durable agent state and exactly-once side-effect records."""

    def __init__(self, db_path: str | Path = "runs/agent_events.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self._create_tables()

    def _create_tables(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    status TEXT NOT NULL,
                    step_count INTEGER NOT NULL DEFAULT 0,
                    termination_reason TEXT,
                    pending_tool_calls TEXT NOT NULL DEFAULT '[]',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    step INTEGER,
                    idempotency_key TEXT UNIQUE,
                    created_at REAL NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_executions (
                    idempotency_key TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    tool_call_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    args_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sent_emails (
                    idempotency_key TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    recipient TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    body TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS run_states (
                    run_id TEXT PRIMARY KEY,
                    current_state TEXT,
                    step_count INTEGER
                )
                """
            )

    def create_run(self, run_id: str, task: str) -> bool:
        now = time.time()
        with self.conn:
            cursor = self.conn.execute(
                """
                INSERT OR IGNORE INTO runs
                    (run_id, task, status, step_count, termination_reason,
                     pending_tool_calls, created_at, updated_at)
                VALUES (?, ?, 'STARTED', 0, NULL, '[]', ?, ?)
                """,
                (run_id, task, now, now),
            )
            if cursor.rowcount == 1:
                self.conn.execute(
                    """
                    INSERT INTO events
                        (run_id, event_type, payload_json, step, idempotency_key, created_at)
                    VALUES (?, 'run_started', ?, 0, NULL, ?)
                    """,
                    (run_id, stable_json({"task": task}), now),
                )
        created = cursor.rowcount == 1
        return created

    def get_run(self, run_id: str) -> RunState | None:
        row = self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return RunState(
            run_id=row["run_id"],
            task=row["task"],
            status=row["status"],
            step_count=row["step_count"],
            termination_reason=row["termination_reason"],
            pending_tool_calls=json.loads(row["pending_tool_calls"] or "[]"),
        )

    def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        step_count: int | None = None,
        termination_reason: str | None = None,
        pending_tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        current = self.get_run(run_id)
        if current is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        next_status = status if status is not None else current.status
        next_step = step_count if step_count is not None else current.step_count
        next_reason = (
            termination_reason if termination_reason is not None else current.termination_reason
        )
        pending = (
            stable_json(pending_tool_calls)
            if pending_tool_calls is not None
            else stable_json(current.pending_tool_calls)
        )
        with self.conn:
            self.conn.execute(
                """
                UPDATE runs
                SET status = ?, step_count = ?, termination_reason = ?,
                    pending_tool_calls = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (next_status, next_step, next_reason, pending, time.time(), run_id),
            )
            self.conn.execute(
                """
                INSERT INTO run_states (run_id, current_state, step_count)
                VALUES (?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    current_state = excluded.current_state,
                    step_count = excluded.step_count
                """,
                (run_id, next_status, next_step),
            )

    def finish_run(self, run_id: str, reason: str, *, step_count: int | None = None) -> None:
        self.update_run(
            run_id,
            status="FINISHED",
            step_count=step_count,
            termination_reason=reason,
            pending_tool_calls=[],
        )
        self.append_event(run_id, "run_finished", {"reason": reason}, step=step_count)

    def append_event(
        self,
        run_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        step: int | None = None,
        idempotency_key: str | None = None,
    ) -> int | None:
        event = Event(
            run_id=run_id,
            event_type=event_type,
            payload=payload or {},
            step=step,
            idempotency_key=idempotency_key,
        )
        try:
            with self.conn:
                cursor = self.conn.execute(
                    """
                    INSERT INTO events
                        (run_id, event_type, payload_json, step, idempotency_key, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.run_id,
                        event.event_type,
                        stable_json(event.payload),
                        event.step,
                        event.idempotency_key,
                        event.created_at,
                    ),
                )
            return int(cursor.lastrowid)
        except sqlite3.IntegrityError:
            return None

    def get_events(self, run_id: str) -> list[Event]:
        rows = self.conn.execute(
            "SELECT * FROM events WHERE run_id = ? ORDER BY id ASC", (run_id,)
        ).fetchall()
        return [
            Event(
                run_id=row["run_id"],
                event_type=row["event_type"],
                payload=json.loads(row["payload_json"]),
                step=row["step"],
                idempotency_key=row["idempotency_key"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def save_event(self, event: Event | EventDTO) -> int | None:
        payload = event.payload
        if isinstance(payload, str):
            try:
                payload = json.loads(payload) if payload else {}
            except json.JSONDecodeError:
                payload = {"text": payload}
        return self.append_event(
            event.run_id,
            event.event_type,
            payload,
            idempotency_key=getattr(event, "idempotency_key", None),
        )

    def log_event(self, event: EventDTO) -> int | None:
        return self.save_event(event)

    def save_state(self, state: RunStateDTO) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO run_states (run_id, current_state, step_count)
                VALUES (?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    current_state = excluded.current_state,
                    step_count = excluded.step_count
                """,
                (state.run_id, state.current_state, state.step_count),
            )

    def get_last_state(self, run_id: str) -> Event | None:
        rows = self.get_events(run_id)
        return rows[-1] if rows else None

    def check_idempotency(self, key: str) -> bool:
        return self.is_key_processed(key)

    def is_key_processed(self, key: str) -> bool:
        tool_row = self.conn.execute(
            "SELECT 1 FROM tool_executions WHERE idempotency_key = ?", (key,)
        ).fetchone()
        email_row = self.conn.execute(
            "SELECT 1 FROM sent_emails WHERE idempotency_key = ?", (key,)
        ).fetchone()
        return tool_row is not None or email_row is not None

    def get_tool_execution(self, idempotency_key: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM tool_executions WHERE idempotency_key = ?", (idempotency_key,)
        ).fetchone()
        if row is None:
            return None
        result = json.loads(row["result_json"]) if row["result_json"] else None
        return {
            "idempotency_key": row["idempotency_key"],
            "run_id": row["run_id"],
            "tool_call_id": row["tool_call_id"],
            "tool_name": row["tool_name"],
            "arguments": json.loads(row["args_json"]),
            "result": result,
            "error": row["error"],
            "status": row["status"],
        }

    def record_tool_execution(
        self,
        *,
        idempotency_key: str,
        run_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any = None,
        error: str | None = None,
        status: str = "completed",
    ) -> None:
        now = time.time()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO tool_executions
                    (idempotency_key, run_id, tool_call_id, tool_name, args_json,
                     result_json, error, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(idempotency_key) DO UPDATE SET
                    result_json = excluded.result_json,
                    error = excluded.error,
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (
                    idempotency_key,
                    run_id,
                    tool_call_id,
                    tool_name,
                    stable_json(arguments),
                    stable_json(result) if result is not None else None,
                    error,
                    status,
                    now,
                    now,
                ),
            )

    def mark_key_processed(self, key: str) -> None:
        self.record_tool_execution(
            idempotency_key=key,
            run_id="unknown",
            tool_call_id=key,
            tool_name="unknown",
            arguments={},
            result={"status": "processed"},
            status="completed",
        )

    def send_email_once(
        self,
        *,
        idempotency_key: str,
        run_id: str,
        to: str,
        subject: str,
        body: str,
    ) -> dict[str, Any]:
        now = time.time()
        with self.conn:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO sent_emails
                    (idempotency_key, run_id, recipient, subject, body, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (idempotency_key, run_id, to, subject, body, now),
            )
        row = self.conn.execute(
            "SELECT * FROM sent_emails WHERE idempotency_key = ?", (idempotency_key,)
        ).fetchone()
        return {
            "status": "queued",
            "idempotency_key": row["idempotency_key"],
            "to": row["recipient"],
            "subject": row["subject"],
            "body": row["body"],
            "created_at": row["created_at"],
        }

    def count_sent_emails(self, run_id: str | None = None) -> int:
        if run_id is None:
            row = self.conn.execute("SELECT COUNT(*) AS count FROM sent_emails").fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) AS count FROM sent_emails WHERE run_id = ?", (run_id,)
            ).fetchone()
        return int(row["count"])

    def close(self) -> None:
        self.conn.close()


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
