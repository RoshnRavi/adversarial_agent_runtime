"""Command line interface for the custom agent runtime."""

from __future__ import annotations

import argparse
import signal
import sys
import uuid

from .config import DEFAULT_CONFIG, RuntimeConfig
from .database import AgentDatabase
from .exceptions import AgentError
from .llm_client import LLMClient
from .loop import AgentLoop
from .memory import MemoryManager
from .replay import replay_run
from .trace import TraceWriter


db_instance: AgentDatabase | None = None


def handle_graceful_shutdown(sig: int, frame: object) -> None:
    print("\nReceived shutdown signal. Closing runtime state...")
    if db_instance is not None:
        db_instance.close()
        print("Database connection closed.")
    raise SystemExit(130)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adversarial agent runtime")
    parser.add_argument("--db", default=str(DEFAULT_CONFIG.db_path), help="SQLite database path")
    parser.add_argument("--server-url", default=DEFAULT_CONFIG.server_url, help="Mock LLM URL")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Start a new task")
    run_parser.add_argument("--task", required=True, help="Task for the agent")

    resume_parser = subparsers.add_parser("resume", help="Resume an existing run")
    resume_parser.add_argument("run_id")

    replay_parser = subparsers.add_parser("replay", help="Replay a recorded trace")
    replay_parser.add_argument("run_id")
    return parser


def _build_loop(config: RuntimeConfig, db: AgentDatabase) -> AgentLoop:
    return AgentLoop(
        db,
        memory=MemoryManager(config.token_budget),
        tracer=TraceWriter(config.trace_dir),
        llm_client=LLMClient(config),
        config=config,
    )


def main(argv: list[str] | None = None) -> int:
    global db_instance

    signal.signal(signal.SIGINT, handle_graceful_shutdown)
    signal.signal(signal.SIGTERM, handle_graceful_shutdown)

    args = build_parser().parse_args(argv)
    config = RuntimeConfig(db_path=args.db, server_url=args.server_url)

    if args.command == "replay":
        try:
            print(replay_run(args.run_id))
            return 0
        except AgentError as exc:
            print(f"replay failed: {exc}", file=sys.stderr)
            return 1

    db_instance = AgentDatabase(config.db_path)
    try:
        loop = _build_loop(config, db_instance)
        if args.command == "run":
            run_id = str(uuid.uuid4())
            print(f"run_id={run_id}")
            state = loop.run_task(run_id, args.task)
        elif args.command == "resume":
            state = loop.resume(args.run_id)
        else:
            raise AssertionError(f"Unhandled command: {args.command}")
        print(
            f"status={state.status} step_count={state.step_count} "
            f"reason={state.termination_reason or ''}"
        )
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if db_instance is not None:
            db_instance.close()
            db_instance = None


if __name__ == "__main__":
    raise SystemExit(main())
