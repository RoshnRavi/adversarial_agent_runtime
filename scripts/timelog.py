"""Small Markdown timelog updater for assessment work."""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

TIMELOG_PATH = Path("TIMELOG.md")
ROW_PATTERN = re.compile(
    r"^\|\s*(?P<date>[^|]+?)\s*\|\s*"
    r"(?P<duration>[0-9]+(?:\.[0-9]+)?)h\s*\|\s*"
    r"(?P<notes>.*)\s*\|$"
)
HEADER_LINES = [
    "# Time Log",
    "",
    "| Date | Duration | Notes |",
    "| --- | ---: | --- |",
]


def rounded_elapsed_hours(elapsed_seconds: float) -> float:
    """Round elapsed wall time up to the nearest 0.05h, with a visible minimum."""

    if elapsed_seconds < 0:
        raise ValueError("elapsed_seconds must be non-negative")
    raw_hours = elapsed_seconds / 3600
    return max(0.05, math.ceil(raw_hours / 0.05) * 0.05)


def record_entry(
    *,
    path: Path = TIMELOG_PATH,
    duration_hours: float,
    note: str,
    entry_date: str | None = None,
) -> None:
    if duration_hours <= 0:
        raise ValueError("duration_hours must be positive")
    day = entry_date or datetime.now(timezone.utc).astimezone().date().isoformat()
    clean_note = _clean_note(note)
    lines = _read_timelog(path)

    for index, line in enumerate(lines):
        parsed = _parse_row(line)
        if parsed is None or parsed["date"] != day:
            continue
        next_duration = float(parsed["duration"]) + duration_hours
        next_notes = _append_note(parsed["notes"], clean_note)
        lines[index] = _format_row(day, next_duration, next_notes)
        _write_timelog(path, lines)
        return

    lines.append(_format_row(day, duration_hours, clean_note))
    _write_timelog(path, lines)


def run_and_record(
    *,
    command: list[str],
    note: str,
    path: Path = TIMELOG_PATH,
    entry_date: str | None = None,
) -> int:
    if not command:
        raise ValueError("command must not be empty")

    started = time.perf_counter()
    try:
        completed = subprocess.run(command, check=False)
        return_code = completed.returncode
        status = "passed" if return_code == 0 else f"failed exit {return_code}"
    except OSError as exc:
        return_code = 127
        status = f"failed {type(exc).__name__}: {exc}"
    elapsed_hours = rounded_elapsed_hours(time.perf_counter() - started)
    record_entry(
        path=path,
        duration_hours=elapsed_hours,
        note=f"{note} ({status})",
        entry_date=entry_date,
    )
    return return_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Update TIMELOG.md for assessment work.")
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    record_parser = subparsers.add_parser("record", help="Record a manual timelog entry.")
    record_parser.add_argument("--duration-hours", required=True, type=_positive_float)
    record_parser.add_argument("--note", required=True)
    record_parser.add_argument("--path", type=Path, default=TIMELOG_PATH)
    record_parser.add_argument("--date", dest="entry_date")

    run_parser = subparsers.add_parser("run", help="Run a command and record elapsed time.")
    run_parser.add_argument("--note", required=True)
    run_parser.add_argument("--path", type=Path, default=TIMELOG_PATH)
    run_parser.add_argument("--date", dest="entry_date")
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command_name == "record":
        record_entry(
            path=args.path,
            duration_hours=args.duration_hours,
            note=args.note,
            entry_date=args.entry_date,
        )
        return 0
    if args.command_name == "run":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            parser.error("run requires a command after --")
        return run_and_record(
            command=command,
            note=args.note,
            path=args.path,
            entry_date=args.entry_date,
        )
    raise AssertionError(f"Unhandled command: {args.command_name}")


def _read_timelog(path: Path) -> list[str]:
    if not path.exists():
        return HEADER_LINES.copy()
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines or HEADER_LINES.copy()


def _write_timelog(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _parse_row(line: str) -> dict[str, str] | None:
    match = ROW_PATTERN.match(line)
    if match is None:
        return None
    return {
        "date": match.group("date"),
        "duration": match.group("duration"),
        "notes": match.group("notes").strip(),
    }


def _format_row(day: str, duration_hours: float, notes: str) -> str:
    return f"| {day} | {_format_duration(duration_hours)} | {notes} |"


def _format_duration(duration_hours: float) -> str:
    return f"{duration_hours:.2f}".rstrip("0").rstrip(".") + "h"


def _append_note(existing: str, note: str) -> str:
    if not existing:
        return note
    return f"{existing}; {note}"


def _clean_note(note: str) -> str:
    return " ".join(note.replace("|", "\\|").split())


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
