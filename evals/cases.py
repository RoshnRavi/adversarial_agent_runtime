"""Named eval cases covering the assessment scenario table."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    id: str
    name: str
    adversarial: bool = False
    expected_failure: bool = False
    reason: str = ""


EVAL_CASES: list[EvalCase] = [
    EvalCase("S01", "happy path single tool call"),
    EvalCase("S02", "malformed tool arguments", adversarial=True),
    EvalCase("S03", "unknown tool and wrong typed argument", adversarial=True),
    EvalCase("S04", "infinite loop bounded termination", adversarial=True),
    EvalCase("S05", "connection reset fails legibly", adversarial=True),
    EvalCase("S06", "429/529 retry path"),
    EvalCase("S07", "tool-result prompt injection stays data", adversarial=True),
    EvalCase("S08", "context budget blow-up terminates", adversarial=True),
    EvalCase("S09", "duplicate tool ids remain idempotent", adversarial=True),
    EvalCase(
        "S10",
        "true parallel tool isolation",
        adversarial=True,
    ),
    EvalCase(
        "S11",
        "model claim checked against tool error",
        adversarial=True,
    ),
    EvalCase(
        "S12",
        "partial interrupted parallel turn recovery",
        adversarial=True,
    ),
    EvalCase("I01", "local mock server HTTP integration"),
    EvalCase(
        "FAIL01",
        "generic false success claim without target",
        adversarial=True,
        expected_failure=True,
        reason="runtime only rejects false success claims tied to a concrete failed tool target",
    ),
    EvalCase(
        "FAIL02",
        "transactional rollback across mixed tool batch",
        adversarial=True,
        expected_failure=True,
        reason="runtime isolates parallel tool outcomes but does not roll back successful sibling side effects",
    ),
]
