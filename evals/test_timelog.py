from __future__ import annotations

import sys
from pathlib import Path

from scripts.timelog import HEADER_LINES, record_entry, run_and_record


def test_record_entry_creates_new_timelog(tmp_path: Path) -> None:
    path = tmp_path / "TIMELOG.md"

    record_entry(path=path, duration_hours=0.25, note="initial work", entry_date="2026-07-29")

    assert path.read_text(encoding="utf-8").splitlines() == [
        *HEADER_LINES,
        "| 2026-07-29 | 0.25h | initial work |",
    ]


def test_record_entry_updates_existing_date(tmp_path: Path) -> None:
    path = tmp_path / "TIMELOG.md"
    path.write_text(
        "\n".join(
            [
                *HEADER_LINES,
                "| 2026-07-29 | 0.25h | initial work |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    record_entry(path=path, duration_hours=0.5, note="tests", entry_date="2026-07-29")

    assert "| 2026-07-29 | 0.75h | initial work; tests |" in path.read_text(
        encoding="utf-8"
    )


def test_record_entry_can_update_notes_with_escaped_pipes(tmp_path: Path) -> None:
    path = tmp_path / "TIMELOG.md"

    record_entry(path=path, duration_hours=0.25, note="read R1 | R2", entry_date="2026-07-29")
    record_entry(path=path, duration_hours=0.25, note="implementation", entry_date="2026-07-29")

    assert "| 2026-07-29 | 0.5h | read R1 \\| R2; implementation |" in path.read_text(
        encoding="utf-8"
    )


def test_record_entry_preserves_header_and_older_rows(tmp_path: Path) -> None:
    path = tmp_path / "TIMELOG.md"
    older_row = "| 2026-07-28 | 2.25h | Implemented runtime. |"
    path.write_text("\n".join([*HEADER_LINES, older_row]) + "\n", encoding="utf-8")

    record_entry(path=path, duration_hours=0.25, note="docs", entry_date="2026-07-29")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[:4] == HEADER_LINES
    assert older_row in lines
    assert "| 2026-07-29 | 0.25h | docs |" in lines


def test_run_and_record_records_success(tmp_path: Path) -> None:
    path = tmp_path / "TIMELOG.md"

    return_code = run_and_record(
        command=[sys.executable, "-c", "raise SystemExit(0)"],
        note="successful command",
        path=path,
        entry_date="2026-07-29",
    )

    assert return_code == 0
    text = path.read_text(encoding="utf-8")
    assert "| 2026-07-29 | 0.05h | successful command (passed) |" in text


def test_run_and_record_records_failure_and_returns_exit_code(tmp_path: Path) -> None:
    path = tmp_path / "TIMELOG.md"

    return_code = run_and_record(
        command=[sys.executable, "-c", "raise SystemExit(7)"],
        note="failing command",
        path=path,
        entry_date="2026-07-29",
    )

    assert return_code == 7
    text = path.read_text(encoding="utf-8")
    assert "| 2026-07-29 | 0.05h | failing command (failed exit 7) |" in text
