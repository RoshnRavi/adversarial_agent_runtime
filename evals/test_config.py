from pathlib import Path
from textwrap import dedent

import pytest

from agent.exceptions import AgentConfigError
from agent.executer import ToolExecutor
from agent.message import MAX_CONTEXT_TOKENS
from agent.response import ToolCall
from agent.run import AgentDatabase
from agent.runtime import DEFAULT_CONFIG, RuntimeConfig, load_runtime_config
from agent.user import build_parser


def _write_config(path: Path, *, http_allow_hosts: str | None = None) -> None:
    hosts = http_allow_hosts or dedent(
        """
        http_allow_hosts:
          - localhost
          - 127.0.0.1
        """
    ).strip()
    path.write_text(
        f"""db_path: runs/custom.db
trace_dir: runs/custom-traces
workspace_root: custom-workspace
server_url: http://localhost:9999/chat
token_budget: 1234
cost_budget_tokens: 4321
max_steps: 7
max_retries: 2
retry_base_delay: 0.5
request_timeout_seconds: 1.5
circuit_max_failures: 4
circuit_cooldown_seconds: 9
no_progress_limit: 2
python_timeout_seconds: 3
python_memory_mb: 32
http_timeout_seconds: 2.5
{hosts}
""",
        encoding="utf-8",
    )


def test_default_config_loads_from_yaml() -> None:
    assert DEFAULT_CONFIG.db_path == Path("runs/agent_events.db")
    assert DEFAULT_CONFIG.trace_dir == Path("runs/traces")
    assert DEFAULT_CONFIG.workspace_root == Path("workspace")
    assert DEFAULT_CONFIG.token_budget == MAX_CONTEXT_TOKENS == 8000
    assert DEFAULT_CONFIG.cost_budget_tokens == 50000
    assert DEFAULT_CONFIG.http_allow_hosts == ("localhost", "127.0.0.1")


def test_load_runtime_config_coerces_paths_and_hosts(tmp_path: Path) -> None:
    config_path = tmp_path / "config_agent.yaml"
    _write_config(config_path)

    config = load_runtime_config(config_path)

    assert config.db_path == Path("runs/custom.db")
    assert config.trace_dir == Path("runs/custom-traces")
    assert config.workspace_root == Path("custom-workspace")
    assert config.http_allow_hosts == ("localhost", "127.0.0.1")
    assert config.cost_budget_tokens == 4321
    assert config.retry_base_delay == 0.5
    assert config.http_timeout_seconds == 2.5


def test_cli_overrides_layer_on_yaml_defaults(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--db",
            str(tmp_path / "cli.db"),
            "--server-url",
            "http://localhost:7777/chat",
            "run",
            "--task",
            "hello",
        ]
    )

    config = load_runtime_config(db_path=args.db, server_url=args.server_url)

    assert config.db_path == tmp_path / "cli.db"
    assert config.server_url == "http://localhost:7777/chat"
    assert config.trace_dir == DEFAULT_CONFIG.trace_dir


def test_missing_config_keys_fail_clearly(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("db_path: runs/custom.db\n", encoding="utf-8")

    with pytest.raises(AgentConfigError, match="missing runtime config key"):
        load_runtime_config(config_path)


def test_invalid_config_values_fail_clearly(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    _write_config(config_path, http_allow_hosts="http_allow_hosts: localhost")

    with pytest.raises(AgentConfigError, match="http_allow_hosts must be a non-empty list"):
        load_runtime_config(config_path)


def test_invalid_cost_budget_fails_clearly(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    _write_config(config_path)
    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(text.replace("cost_budget_tokens: 4321", "cost_budget_tokens: 0"), encoding="utf-8")

    with pytest.raises(AgentConfigError, match="cost_budget_tokens must be a positive integer"):
        load_runtime_config(config_path)


def test_token_budget_above_hard_ceiling_fails_clearly(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    _write_config(config_path)
    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        text.replace("token_budget: 1234", "token_budget: 8001"),
        encoding="utf-8",
    )

    with pytest.raises(AgentConfigError, match="token_budget must be <= 8000"):
        load_runtime_config(config_path)


def test_executor_uses_configured_workspace_root(tmp_path: Path) -> None:
    config = RuntimeConfig(
        db_path=tmp_path / "agent.db",
        trace_dir=tmp_path / "traces",
        workspace_root=tmp_path / "workspace",
    )
    db = AgentDatabase(config.db_path)
    executor = ToolExecutor(db, run_id="configured-workspace", config=config)

    write_result = executor.execute(
        ToolCall(
            id="write",
            name="write_file",
            arguments={"path": "nested/out.txt", "content": "ok"},
        ),
        "write-key",
    )
    python_result = executor.execute(
        ToolCall(
            id="python",
            name="run_python",
            arguments={"code": "from pathlib import Path; print(Path.cwd())"},
        ),
        "python-key",
    )

    assert write_result.ok
    assert (config.workspace_root / "nested/out.txt").read_text(encoding="utf-8") == "ok"
    assert python_result.ok
    assert str(config.workspace_root.resolve()) in python_result.result["stdout"]
    db.close()
