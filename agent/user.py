"""User-facing entrypoint for the custom agent runtime."""

from __future__ import annotations

import argparse
import signal
import sys

from .exceptions import AgentError
from .run import AgentDatabase, replay_run
from .runtime import DEFAULT_CONFIG, AgentRuntime, RuntimeConfig, load_runtime_config

db_instance: AgentDatabase | None = None


def handle_graceful_shutdown(sig: int, frame: object) -> None:
    print("\nReceived shutdown signal. Closing runtime state...")
    if db_instance is not None:
        db_instance.close()
        print("Database connection closed.")
    raise SystemExit(130)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adversarial agent runtime")
    parser.add_argument(
        "--db",
        default=None,
        help=f"SQLite database path. Default: {DEFAULT_CONFIG.db_path}",
    )
    parser.add_argument(
        "--server-url",
        default=None,
        help=f"Mock LLM URL. Default: {DEFAULT_CONFIG.server_url}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Start a new task")
    run_parser.add_argument("--run-id", help="Optional deterministic run id")
    run_parser.add_argument("--task", required=True, help="Task for the agent")

    resume_parser = subparsers.add_parser("resume", help="Resume an existing run")
    resume_parser.add_argument("run_id")

    replay_parser = subparsers.add_parser("replay", help="Replay a recorded trace")
    replay_parser.add_argument("run_id")
    return parser


def _build_runtime(config: RuntimeConfig, db: AgentDatabase) -> AgentRuntime:
    return AgentRuntime(db, config=config)


def main(argv: list[str] | None = None) -> int:
    global db_instance

    signal.signal(signal.SIGINT, handle_graceful_shutdown)
    signal.signal(signal.SIGTERM, handle_graceful_shutdown)

    args = build_parser().parse_args(argv)
    config = load_runtime_config(db_path=args.db, server_url=args.server_url)

    if args.command == "replay":
        try:
            # Replay is intentionally available without constructing the runtime
            # or opening a model connection.
            print(replay_run(args.run_id))
            return 0
        except AgentError as exc:
            print(f"replay failed: {exc}", file=sys.stderr)
            return 1

    db_instance = AgentDatabase(config.db_path)
    try:
        runtime = _build_runtime(config, db_instance)
        if args.command == "run":
            state = (
                runtime.run_task(args.run_id, args.task)
                if args.run_id
                else runtime.start_task(args.task)
            )
            print(f"run_id={state.run_id}")
        elif args.command == "resume":
            state = runtime.resume(args.run_id)
        else:
            raise AssertionError(f"Unhandled command: {args.command}")
        print(
            f"status={state.status} step_count={state.step_count} "
            f"reason={state.termination_reason or ''}"
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - report unexpected failures at the CLI boundary.
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if db_instance is not None:
            db_instance.close()
            db_instance = None


if __name__ == "__main__":
    raise SystemExit(main())
