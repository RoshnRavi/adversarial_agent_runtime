"""Tests for chaos-harness helpers and email-count assertions."""

from pathlib import Path

from agent.run import AgentDatabase
from agent.user import build_parser
from harness.chaos import assert_sent_email_count, build_agent_command, run_exists


def test_user_parser_accepts_run_id_for_run() -> None:
    args = build_parser().parse_args(["run", "--run-id", "r2-run", "--task", "hello"])

    assert args.command == "run"
    assert args.run_id == "r2-run"
    assert args.task == "hello"


def test_chaos_builds_run_and_resume_commands() -> None:
    run_command = build_agent_command(
        db_path="runs/chaos.db",
        server_url="http://localhost:8000/chat",
        run_id="chaos-r2",
        task="send exactly one email",
        resume=False,
        python_executable="python3",
    )
    resume_command = build_agent_command(
        db_path="runs/chaos.db",
        server_url="http://localhost:8000/chat",
        run_id="chaos-r2",
        task="send exactly one email",
        resume=True,
        python_executable="python3",
    )

    assert run_command == [
        "python3",
        "-m",
        "agent.user",
        "--db",
        "runs/chaos.db",
        "--server-url",
        "http://localhost:8000/chat",
        "run",
        "--run-id",
        "chaos-r2",
        "--task",
        "send exactly one email",
    ]
    assert resume_command == [
        "python3",
        "-m",
        "agent.user",
        "--db",
        "runs/chaos.db",
        "--server-url",
        "http://localhost:8000/chat",
        "resume",
        "chaos-r2",
    ]


def test_chaos_email_count_assertion_detects_mismatch(tmp_path: Path) -> None:
    db_path = tmp_path / "agent.db"
    db = AgentDatabase(db_path)
    db.send_email_once(
        idempotency_key="email-key",
        run_id="chaos-r2",
        to="a@example.com",
        subject="Hi",
        body="Once",
    )
    db.close()

    assert assert_sent_email_count(db_path, "chaos-r2", 1)
    assert not assert_sent_email_count(db_path, "chaos-r2", 2)


def test_chaos_run_exists_handles_missing_and_present_runs(tmp_path: Path) -> None:
    db_path = tmp_path / "agent.db"

    assert not run_exists(db_path, "chaos-r2")

    db = AgentDatabase(db_path)
    db.create_run("chaos-r2", "send exactly one email")
    db.close()

    assert run_exists(db_path, "chaos-r2")
