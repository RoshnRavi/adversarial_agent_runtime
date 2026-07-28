"""Runtime configuration for the custom agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    db_path: Path = Path("runs/agent_events.db")
    trace_dir: Path = Path("runs/traces")
    workspace_root: Path = Path("workspace")
    server_url: str = "http://localhost:8000/chat"
    token_budget: int = 8000
    max_steps: int = 30
    max_retries: int = 4
    retry_base_delay: float = 0.25
    request_timeout_seconds: float = 5.0
    circuit_max_failures: int = 5
    circuit_cooldown_seconds: int = 60
    no_progress_limit: int = 3
    python_timeout_seconds: int = 5
    python_memory_mb: int = 64
    http_allow_hosts: tuple[str, ...] = field(default_factory=lambda: ("localhost", "127.0.0.1"))


DEFAULT_CONFIG = RuntimeConfig()
