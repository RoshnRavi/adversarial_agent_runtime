"""Chaos runner that repeatedly kills and resumes agent runs."""

from __future__ import annotations

import argparse
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.run import AgentDatabase


def build_agent_command(
    *,
    db_path: str,
    server_url: str,
    run_id: str,
    task: str,
    resume: bool,
    python_executable: str = sys.executable,
) -> list[str]:
    base = [
        python_executable,
        "-m",
        "agent.user",
        "--db",
        db_path,
        "--server-url",
        server_url,
    ]
    if resume:
        return [*base, "resume", run_id]
    return [*base, "run", "--run-id", run_id, "--task", task]


def run_chaos(command: list[str], *, attempts: int, min_delay: float, max_delay: float) -> int:
    failures = 0
    for _ in range(attempts):
        process = subprocess.Popen(command)
        time.sleep(random.uniform(min_delay, max_delay))
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        failures += process.returncode not in (0, -signal.SIGTERM)
    return failures


def run_agent_chaos(
    *,
    attempts: int,
    min_delay: float,
    max_delay: float,
    db_path: str,
    server_url: str,
    run_id: str,
    task: str,
    expected_email_count: int | None = None,
) -> int:
    failures = 0
    for attempt in range(attempts):
        resume = attempt > 0 and run_exists(db_path, run_id)
        command = build_agent_command(
            db_path=db_path,
            server_url=server_url,
            run_id=run_id,
            task=task,
            resume=resume,
        )
        process = subprocess.Popen(command)
        time.sleep(random.uniform(min_delay, max_delay))
        if process.poll() is None:
            process.kill()
            process.wait()
        failures += process.returncode not in (0, -signal.SIGKILL)

    final = subprocess.run(
        build_agent_command(
            db_path=db_path,
            server_url=server_url,
            run_id=run_id,
            task=task,
            resume=True,
        ),
        check=False,
    )
    failures += final.returncode != 0
    if expected_email_count is not None:
        failures += int(not assert_sent_email_count(db_path, run_id, expected_email_count))
    return failures


def assert_sent_email_count(db_path: str | Path, run_id: str, expected_count: int) -> bool:
    db = AgentDatabase(db_path)
    try:
        return db.count_sent_emails(run_id) == expected_count
    finally:
        db.close()


def run_exists(db_path: str | Path, run_id: str) -> bool:
    path = Path(db_path)
    if not path.exists():
        return False
    db = AgentDatabase(path)
    try:
        return db.get_run(run_id) is not None
    finally:
        db.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repeatedly run and randomly kill a command.")
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--min-delay", type=float, default=0.05)
    parser.add_argument("--max-delay", type=float, default=1.0)
    parser.add_argument("--db", dest="db_path")
    parser.add_argument("--server-url", default="http://localhost:8000/chat")
    parser.add_argument("--run-id")
    parser.add_argument("--task")
    parser.add_argument("--assert-email-count", type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.run_id or args.task or args.db_path:
        if not args.run_id or not args.task or not args.db_path:
            raise SystemExit("--db, --run-id, and --task are required for agent chaos mode")
        return run_agent_chaos(
            attempts=args.attempts,
            min_delay=args.min_delay,
            max_delay=args.max_delay,
            db_path=args.db_path,
            server_url=args.server_url,
            run_id=args.run_id,
            task=args.task,
            expected_email_count=args.assert_email_count,
        )
    if not args.command:
        raise SystemExit("usage: chaos.py [options] -- <command...>")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return run_chaos(
        command,
        attempts=args.attempts,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
    )


if __name__ == "__main__":
    raise SystemExit(main())
