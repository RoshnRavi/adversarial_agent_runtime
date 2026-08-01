"""Tests for parsing mock model text, tool calls, and partial turns."""

from agent.response import parse_agent_response


def test_parse_final_text_response() -> None:
    parsed = parse_agent_response({"content": "done"}, step=1)

    assert parsed.assistant_text == "done"
    assert parsed.tool_calls == []
    assert parsed.parse_errors == []


def test_parse_single_tool_call() -> None:
    parsed = parse_agent_response(
        {
            "tool_call": {
                "id": "write-1",
                "name": "write_file",
                "arguments": {"path": "a.txt", "content": "ok"},
            }
        },
        step=2,
    )

    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].id == "write-1"
    assert parsed.tool_calls[0].name == "write_file"
    assert parsed.tool_calls[0].arguments == {"path": "a.txt", "content": "ok"}
    assert parsed.parse_errors == []


def test_parse_tool_calls_list() -> None:
    parsed = parse_agent_response(
        {
            "tool_calls": [
                {"id": "one", "name": "read_file", "arguments": {"path": "a.txt"}},
                {"id": "two", "name": "run_python", "arguments": {"code": "print(1)"}},
            ]
        },
        step=3,
    )

    assert [call.id for call in parsed.tool_calls] == ["one", "two"]
    assert [call.name for call in parsed.tool_calls] == ["read_file", "run_python"]
    assert parsed.parse_errors == []


def test_parse_tool_use_list() -> None:
    parsed = parse_agent_response(
        {"tool_use": [{"id": "use-1", "name": "read_file", "input": {"path": "a.txt"}}]},
        step=4,
    )

    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].id == "use-1"
    assert parsed.tool_calls[0].arguments == {"path": "a.txt"}
    assert parsed.parse_errors == []


def test_parse_content_list_tool_calls_and_text() -> None:
    parsed = parse_agent_response(
        {
            "content": [
                {"type": "text", "text": "checking file"},
                {"type": "tool_use", "id": "read-1", "name": "read_file", "input": {"path": "a.txt"}},
            ]
        },
        step=5,
    )

    assert parsed.assistant_text == "checking file"
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].id == "read-1"
    assert parsed.tool_calls[0].name == "read_file"
    assert parsed.parse_errors == []


def test_parse_json_encoded_tool_call_inside_content() -> None:
    parsed = parse_agent_response(
        {
            "content": (
                '{"tool_call":{"id":"json-1","name":"run_python",'
                '"arguments":{"code":"print(1)"}}}'
            )
        },
        step=6,
    )

    assert parsed.tool_calls[0].id == "json-1"
    assert parsed.tool_calls[0].name == "run_python"
    assert parsed.tool_calls[0].arguments == {"code": "print(1)"}
    assert parsed.parse_errors == []


def test_parse_malformed_non_object_tool_item() -> None:
    parsed = parse_agent_response({"tool_calls": ["bad"]}, step=7)

    assert parsed.tool_calls == []
    assert parsed.parse_errors == ["tool call 0 is not an object"]


def test_parse_missing_tool_name() -> None:
    parsed = parse_agent_response({"tool_call": {"id": "missing", "arguments": {}}}, step=8)

    assert parsed.tool_calls == []
    assert parsed.parse_errors == ["tool call 0 is missing a valid name"]


def test_parse_malformed_arguments_string_records_error() -> None:
    parsed = parse_agent_response(
        {"tool_call": {"id": "bad-args", "name": "write_file", "arguments": "{bad"}},
        step=9,
    )

    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].arguments == {
        "_malformed_arguments": "{bad",
        "_error": "malformed tool arguments",
    }
    assert parsed.parse_errors == ["malformed tool arguments"]


def test_parse_missing_tool_id_falls_back_to_step_index() -> None:
    parsed = parse_agent_response(
        {"tool_call": {"name": "read_file", "arguments": {"path": "a.txt"}}},
        step=10,
    )

    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].id == "step-10-tool-0"
    assert parsed.parse_errors == []


def test_parse_partial_tool_turn_metadata() -> None:
    parsed = parse_agent_response(
        {
            "finish_reason": "tool_calls_interrupted",
            "expected_tool_call_count": 3,
            "tool_calls": [
                {"id": "one", "name": "write_file", "arguments": {"path": "a.txt", "content": "x"}}
            ],
        },
        step=11,
    )

    assert parsed.partial_turn
    assert parsed.expected_tool_call_count == 3
    assert len(parsed.tool_calls) == 1
