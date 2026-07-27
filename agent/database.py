import sqlite3
import json
from dataclasses import dataclass
from typing import Optional

@dataclass
class RunStateDTO:
    """DTO for the current state of a run."""
    run_id: str
    current_state: str
    step_count: int

@dataclass
class EventDTO:
    """DTO for a single event log."""
    run_id: str
    event_type: str
    payload: str
    idempotency_key: str

class AgentDatabase:
    """Manages the SQLite event log and state tracking."""
    
    def __init__(self, db_path: str = "runs/agent_events.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        # WAL mode makes writes faster and safer during crashes
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._create_tables()

    def _create_tables(self):
        """Creates the necessary tables if they do not exist."""
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS run_states (
                    run_id TEXT PRIMARY KEY,
                    current_state TEXT,
                    step_count INTEGER
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    event_type TEXT,
                    payload TEXT,
                    idempotency_key TEXT UNIQUE
                )
            """)

    def save_state(self, state: RunStateDTO):
        """Saves the current state of the Finite State Machine."""
        with self.conn:
            self.conn.execute("""
                INSERT INTO run_states (run_id, current_state, step_count)
                VALUES (?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                current_state=excluded.current_state,
                step_count=excluded.step_count
            """, (state.run_id, state.current_state, state.step_count))

    def check_idempotency(self, key: str) -> bool:
        """Checks if an event (like send_email) already happened."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM events WHERE idempotency_key = ?", (key,))
        return cursor.fetchone() is not None

    def log_event(self, event: EventDTO):
        """Saves an event to the append-only log."""
        if self.check_idempotency(event.idempotency_key):
            return # Do not save duplicate events
            
        with self.conn:
            self.conn.execute("""
                INSERT INTO events (run_id, event_type, payload, idempotency_key)
                VALUES (?, ?, ?, ?)
            """, (event.run_id, event.event_type, event.payload, event.idempotency_key))

    def close(self):
        """Closes the database connection safely."""
        self.conn.close()