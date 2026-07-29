import json
from dataclasses import replace
from pathlib import Path

import pytest

from agent.exceptions import NetworkFailureError
from evals.cases import EVAL_CASES
from evals.runner import EvalInputError, load_eval_inputs, main, run_case


def test_load_eval_inputs_s01_script() -> None:
    inputs = load_eval_inputs()

    s01 = inputs["S01"]

    assert s01.run_id == "eval-s01"
    assert s01.task == "write a file"
    assert len(s01.responses) == 2
    assert s01.responses[0]["tool_call"]["name"] == "write_file"
    assert s01.responses[1] == {"content": "done"}


def test_load_eval_inputs_s05_error_marker() -> None:
    response = load_eval_inputs()["S05"].responses[0]

    assert isinstance(response, NetworkFailureError)
    assert str(response) == "connection reset mid-response"


def test_load_eval_inputs_s08_token_budget_and_content() -> None:
    s08 = load_eval_inputs()["S08"]

    assert s08.token_budget == 50
    assert len(s08.responses[0]["content"].split()) == 500


def test_all_eval_cases_have_yaml_inputs() -> None:
    inputs = load_eval_inputs()
    scripted_case_ids = {case.id for case in EVAL_CASES if case.id != "I01"}

    assert scripted_case_ids <= set(inputs)


def test_expected_failure_cases_are_executed() -> None:
    inputs = load_eval_inputs()
    f01 = next(case for case in EVAL_CASES if case.id == "F01")
    custom_inputs = {
        "F01": replace(
            inputs["F01"],
            responses=[{"content": "done without a failed tool result"}],
        )
    }

    passed, detail = run_case(f01, custom_inputs)

    assert not passed
    assert detail == f01.reason


@pytest.mark.parametrize("case_id", ["S10", "S11", "S12"])
def test_s10_to_s12_cases_now_pass(case_id: str) -> None:
    case = next(case for case in EVAL_CASES if case.id == case_id)

    passed, detail = run_case(case)

    assert passed
    assert detail == ""


@pytest.mark.parametrize("case_id", ["F01", "F02"])
def test_expected_failure_cases_return_documented_reasons(case_id: str) -> None:
    case = next(case for case in EVAL_CASES if case.id == case_id)

    passed, detail = run_case(case)

    assert not passed
    assert detail == case.reason


def test_eval_runner_summary_matches_baseline(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main()
    output = capsys.readouterr().out.splitlines()
    summary = json.loads(next(line for line in output if line.startswith("{")))
    baseline = json.loads(Path("evals/baseline.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert summary == baseline


def test_missing_required_eval_input_field_fails_clearly(tmp_path: Path) -> None:
    input_path = tmp_path / "input.yaml"
    input_path.write_text(
        """cases:
  S01:
    run_id: eval-s01
    responses:
      - content: done
""",
        encoding="utf-8",
    )

    with pytest.raises(EvalInputError, match="S01: task must be a non-empty string"):
        load_eval_inputs(input_path)
