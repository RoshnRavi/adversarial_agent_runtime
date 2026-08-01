"""Custom eval runner that prints pass rate and baseline diff."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agent.exceptions import NetworkFailureError
from agent.message import MemoryManager
from agent.run import AgentDatabase, TraceWriter
from agent.runtime import AgentRuntime, RuntimeConfig
from mockllm.server import build_server

from .cases import EVAL_CASES, EvalCase

INPUT_PATH = Path(__file__).with_name("input.yaml")
DEFAULT_TOKEN_BUDGET = 8000


@dataclass(frozen=True)
class EvalInput:
    run_id: str
    task: str
    responses: list[Any]
    token_budget: int = DEFAULT_TOKEN_BUDGET
    runtime_overrides: dict[str, Any] = field(default_factory=dict)
    expected_tool_call_count: int | None = None


class EvalInputError(ValueError):
    """Raised when evals/input.yaml cannot be loaded into scripted eval inputs."""


class ScriptedLLM:
    """Deterministic model double used for fast non-HTTP eval cases."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.last_retries: list[Any] = []

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.responses:
            return {"content": "done"}
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _loop(
    tmp: Path,
    responses: list[Any],
    *,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    runtime_overrides: dict[str, Any] | None = None,
) -> tuple[AgentRuntime, AgentDatabase]:
    config = RuntimeConfig(
        db_path=tmp / "agent.db",
        trace_dir=tmp / "traces",
        workspace_root=tmp / "workspace",
        token_budget=token_budget,
        retry_base_delay=0.0,
        **(runtime_overrides or {}),
    )
    db = AgentDatabase(config.db_path)
    runtime = AgentRuntime(
        db,
        memory=MemoryManager(config.token_budget),
        tracer=TraceWriter(config.trace_dir),
        llm_client=ScriptedLLM(responses),
        config=config,
    )
    return runtime, db


def load_eval_inputs(path: Path | None = None) -> dict[str, EvalInput]:
    input_path = INPUT_PATH if path is None else path
    try:
        loaded = yaml.safe_load(input_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvalInputError(f"failed to read eval input file: {input_path}") from exc
    except yaml.YAMLError as exc:
        raise EvalInputError(f"failed to parse eval input YAML: {exc}") from exc

    if not isinstance(loaded, dict):
        raise EvalInputError("eval input file must contain a mapping")
    raw_cases = loaded.get("cases")
    if not isinstance(raw_cases, dict):
        raise EvalInputError("eval input file must contain a cases mapping")

    inputs: dict[str, EvalInput] = {}
    for case_id, raw_case in raw_cases.items():
        if not isinstance(case_id, str) or not case_id:
            raise EvalInputError("eval case id must be a non-empty string")
        if not isinstance(raw_case, dict):
            raise EvalInputError(f"{case_id}: case input must be a mapping")

        raw_responses = raw_case.get("responses")
        if not isinstance(raw_responses, list) or not raw_responses:
            raise EvalInputError(f"{case_id}: responses must be a non-empty list")

        token_budget = raw_case.get("token_budget", DEFAULT_TOKEN_BUDGET)
        if type(token_budget) is not int or token_budget <= 0:
            raise EvalInputError(f"{case_id}: token_budget must be a positive integer")

        inputs[case_id] = EvalInput(
            run_id=_required_str(raw_case, "run_id", case_id),
            task=_required_str(raw_case, "task", case_id),
            responses=[
                _decode_response(case_id, index, item)
                for index, item in enumerate(raw_responses)
            ],
            token_budget=token_budget,
            runtime_overrides=_runtime_overrides(raw_case, case_id),
            expected_tool_call_count=_optional_positive_int(
                raw_case,
                "expected_tool_call_count",
                case_id,
            ),
        )
    return inputs


def _required_str(raw_case: dict[str, Any], key: str, case_id: str) -> str:
    value = raw_case.get(key)
    if not isinstance(value, str) or not value:
        raise EvalInputError(f"{case_id}: {key} must be a non-empty string")
    return value


def _decode_response(case_id: str, index: int, item: Any) -> Any:
    if not isinstance(item, dict):
        raise EvalInputError(f"{case_id}: response {index} must be a mapping")
    if "error" not in item:
        return item

    error = item["error"]
    if not isinstance(error, dict):
        raise EvalInputError(f"{case_id}: response {index} error must be a mapping")
    error_type = error.get("type")
    message = error.get("message")
    if error_type != "NetworkFailureError" or not isinstance(message, str):
        raise EvalInputError(
            f"{case_id}: response {index} has unsupported error marker {error_type!r}"
        )
    return NetworkFailureError(message)


def _runtime_overrides(raw_case: dict[str, Any], case_id: str) -> dict[str, Any]:
    raw_overrides = raw_case.get("runtime", {})
    if not isinstance(raw_overrides, dict):
        raise EvalInputError(f"{case_id}: runtime must be a mapping")
    allowed = {"python_timeout_seconds"}
    unknown = sorted(set(raw_overrides) - allowed)
    if unknown:
        raise EvalInputError(
            f"{case_id}: unsupported runtime override(s): {', '.join(unknown)}"
        )
    overrides: dict[str, Any] = {}
    for key, value in raw_overrides.items():
        if type(value) is not int or value <= 0:
            raise EvalInputError(f"{case_id}: runtime.{key} must be a positive integer")
        overrides[key] = value
    return overrides


def _optional_positive_int(
    raw_case: dict[str, Any],
    key: str,
    case_id: str,
) -> int | None:
    if key not in raw_case:
        return None
    value = raw_case[key]
    if type(value) is not int or value <= 0:
        raise EvalInputError(f"{case_id}: {key} must be a positive integer")
    return value


def run_case(case: EvalCase, inputs: dict[str, EvalInput] | None = None) -> tuple[bool, str]:
    # Integration cases use the real HTTP client against a local server; the
    # rest use ScriptedLLM so failures stay deterministic and cheap.
    if case.id == "I01":
        return _run_local_mock_server_case()
    if case.id == "I02":
        return _run_live_s04_case()

    eval_inputs = load_eval_inputs() if inputs is None else inputs
    eval_input = eval_inputs.get(case.id)
    if eval_input is None:
        return False, "case not implemented"

    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        loop, db = _loop(
            tmp,
            eval_input.responses,
            token_budget=eval_input.token_budget,
            runtime_overrides=eval_input.runtime_overrides,
        )
        try:
            state = loop.run_task(eval_input.run_id, eval_input.task)

            if case.id == "S01":
                return state.status == "FINISHED", state.termination_reason or ""

            if case.id == "S02":
                return state.status == "FINISHED", state.termination_reason or ""

            if case.id == "S03":
                return state.status == "FINISHED", state.termination_reason or ""

            if case.id == "S04":
                return "NoProgressError" in (state.termination_reason or ""), state.termination_reason or ""

            if case.id == "S05":
                return "NetworkFailureError" in (state.termination_reason or ""), state.termination_reason or ""

            if case.id == "S06":
                return state.status == "FINISHED", state.termination_reason or ""

            if case.id == "S07":
                passed = db.count_sent_emails(eval_input.run_id) == 0 and state.status == "FINISHED"
                return passed, "" if passed else "email count changed"

            if case.id == "S08":
                return "ContextLimitExceededError" in (state.termination_reason or ""), state.termination_reason or ""

            if case.id == "S09":
                passed = db.count_sent_emails(eval_input.run_id) == 1 and state.status == "FINISHED"
                return passed, "" if passed else "email count was not one"

            if case.id == "S10":
                passed = _all_parallel_starts_precede_results(db, eval_input.run_id)
                return passed, "" if passed else case.reason

            if case.id == "S11":
                failed_tool_result = _has_failed_tool_result(db, eval_input.run_id)
                passed = failed_tool_result and "ModelContradictionError" in (
                    state.termination_reason or ""
                )
                return passed, "" if passed else state.termination_reason or case.reason

            if case.id == "S12":
                expected_count = eval_input.expected_tool_call_count or 0
                actual_count = _tool_execution_started_count(db, eval_input.run_id)
                passed = (
                    expected_count
                    and actual_count == 0
                    and "PartialToolTurnError" in (state.termination_reason or "")
                )
                return passed, "" if passed else state.termination_reason or case.reason

            if case.id == "A13":
                failed_tool_result = _has_failed_tool_result(db, eval_input.run_id)
                passed = failed_tool_result and "ModelContradictionError" in (
                    state.termination_reason or ""
                )
                return passed, "" if passed else state.termination_reason or case.reason

            if case.id == "A14":
                side_effect_path = loop.config.workspace_root / "f02-side-effect.txt"
                passed = not side_effect_path.exists()
                return passed, "" if passed else case.reason

            if case.id == "FAIL01":
                side_effect_path = loop.config.workspace_root / "py-side-effect.txt"
                passed = not side_effect_path.exists()
                return passed, "" if passed else case.reason

            if case.id == "FAIL02":
                passed = db.count_sent_emails(eval_input.run_id) == 0
                return passed, "" if passed else case.reason

            return False, "case not implemented"
        finally:
            db.close()


def _run_local_mock_server_case() -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        server = build_server("127.0.0.1", 0, scenario="S01")
        host, port = server.server_address
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            config = RuntimeConfig(
                db_path=tmp / "agent.db",
                trace_dir=tmp / "traces",
                workspace_root=tmp / "workspace",
                server_url=f"http://{host}:{port}/chat",
                retry_base_delay=0.0,
            )
            db = AgentDatabase(config.db_path)
            try:
                runtime = AgentRuntime(
                    db,
                    memory=MemoryManager(config.token_budget),
                    tracer=TraceWriter(config.trace_dir),
                    config=config,
                )
                state = runtime.run_task("eval-i01", "exercise local mock server S01")
                output_path = config.workspace_root / "mock_s01.txt"
                passed = (
                    state.status == "FINISHED"
                    and output_path.exists()
                    and output_path.read_text(encoding="utf-8") == "ok"
                )
                detail = state.termination_reason or ""
                return passed, "" if passed else detail or "local mock server S01 failed"
            finally:
                db.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


_LIVE_S04_SCRIPT = """
import sys
from pathlib import Path

from agent.message import MemoryManager
from agent.run import AgentDatabase, TraceWriter
from agent.runtime import AgentRuntime, RuntimeConfig

tmp = Path(sys.argv[1])
config = RuntimeConfig(
    db_path=tmp / "agent.db",
    trace_dir=tmp / "traces",
    workspace_root=tmp / "workspace",
    server_url=sys.argv[2],
    retry_base_delay=0.0,
)
db = AgentDatabase(config.db_path)
try:
    runtime = AgentRuntime(
        db,
        memory=MemoryManager(config.token_budget),
        tracer=TraceWriter(config.trace_dir),
        config=config,
    )
    state = runtime.run_task("eval-i02", "exercise local mock server S04")
    print(f"{state.status}|{state.step_count}|{state.termination_reason or ''}")
finally:
    db.close()
"""


def _run_live_s04_case() -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        server = build_server("127.0.0.1", 0, scenario="S04")
        host, port = server.server_address
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        _LIVE_S04_SCRIPT,
                        str(tmp),
                        f"http://{host}:{port}/chat",
                    ],
                    text=True,
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return False, "live S04 timed out"

            db = AgentDatabase(tmp / "agent.db")
            try:
                state = db.get_run("eval-i02")
                if state is None:
                    return False, "live S04 did not create a run"
                reason = state.termination_reason or ""
                passed = (
                    completed.returncode == 0
                    and state.status == "FINISHED"
                    and "NoProgressError" in reason
                )
                detail = reason or completed.stderr.strip() or completed.stdout.strip()
                return passed, "" if passed else detail
            finally:
                db.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


def _all_parallel_starts_precede_results(db: AgentDatabase, run_id: str) -> bool:
    events = db.get_events(run_id)
    start_indexes = [
        index
        for index, event in enumerate(events)
        if event.event_type == "tool_execution_started"
    ]
    result_indexes = [
        index
        for index, event in enumerate(events)
        if event.event_type == "tool_result"
    ]
    return bool(start_indexes and result_indexes) and max(start_indexes) < min(result_indexes)


def _has_failed_tool_result(db: AgentDatabase, run_id: str) -> bool:
    return any(
        event.event_type == "tool_result" and event.payload.get("ok") is False
        for event in db.get_events(run_id)
    )


def _tool_execution_started_count(db: AgentDatabase, run_id: str) -> int:
    return sum(
        1
        for event in db.get_events(run_id)
        if event.event_type == "tool_execution_started"
    )


def main() -> int:
    inputs = load_eval_inputs()
    results = []
    for case in EVAL_CASES:
        passed, detail = run_case(case, inputs)
        # Expected failures are executed and reported, not skipped, so they still
        # prove the known gap exists and remains documented.
        status = "PASS" if passed else ("EXPECTED_FAIL" if case.expected_failure else "FAIL")
        results.append({"id": case.id, "status": status, "detail": detail})
        print(f"{case.id} {status} - {case.name}" + (f" ({detail})" if detail else ""))

    total = len(results)
    passed_count = sum(1 for result in results if result["status"] == "PASS")
    expected_failures = sum(1 for result in results if result["status"] == "EXPECTED_FAIL")
    unexpected_failures = [result for result in results if result["status"] == "FAIL"]
    summary = {
        "version": 1,
        "total": total,
        "passed": passed_count,
        "expected_failures": expected_failures,
        "unexpected_failures": len(unexpected_failures),
        "pass_rate": round(passed_count / total, 3),
    }
    print(json.dumps(summary, sort_keys=True))
    print(_baseline_diff(summary))
    return 1 if unexpected_failures else 0


def _baseline_diff(summary: dict[str, Any]) -> str:
    baseline_path = Path(__file__).with_name("baseline.json")
    if not baseline_path.exists():
        return "baseline: missing"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    diffs = []
    for key in ("total", "passed", "expected_failures", "unexpected_failures", "pass_rate"):
        if baseline.get(key) != summary.get(key):
            diffs.append(f"{key}: baseline={baseline.get(key)} current={summary.get(key)}")
    return "baseline diff: none" if not diffs else "baseline diff: " + "; ".join(diffs)


if __name__ == "__main__":
    raise SystemExit(main())
