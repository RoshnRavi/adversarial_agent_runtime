"""Chaos runner that repeatedly kills a child process at random points."""

from __future__ import annotations

import argparse
import random
import signal
import subprocess
import time


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repeatedly run and randomly kill a command.")
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--min-delay", type=float, default=0.05)
    parser.add_argument("--max-delay", type=float, default=1.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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

