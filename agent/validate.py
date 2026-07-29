"""Mock LLM transport and per-turn response validation."""

from __future__ import annotations

import hashlib
import json
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import TYPE_CHECKING, Any, Protocol

from .exceptions import CircuitBreakerOpenError, NetworkFailureError
from .response import ToolCall, parse_agent_response
from .runtime import DEFAULT_CONFIG, RuntimeConfig, stable_json

if TYPE_CHECKING:
    from .message import MemoryManager
    from .run import AgentDatabase, TraceWriter


@dataclass
class RetryEvent:
    attempt: int
    reason: str
    delay_seconds: float


class CircuitBreaker:
    def __init__(
        self,
        *,
        max_failures: int = DEFAULT_CONFIG.circuit_max_failures,
        cooldown_seconds: int = DEFAULT_CONFIG.circuit_cooldown_seconds,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.max_failures = max_failures
        self.cooldown_seconds = cooldown_seconds
        self.clock = clock
        self.failure_count = 0
        self.opened_at = 0.0

    def check(self) -> None:
        if self.failure_count < self.max_failures:
            return
        if self.clock() - self.opened_at < self.cooldown_seconds:
            raise CircuitBreakerOpenError("circuit breaker is open")
        self.failure_count = 0

    def record_success(self) -> None:
        self.failure_count = 0
        self.opened_at = 0.0

    def record_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.max_failures:
            self.opened_at = self.clock()


class LLMClient:
    def __init__(
        self,
        config: RuntimeConfig = DEFAULT_CONFIG,
        *,
        sleeper: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self.config = config
        self.sleeper = sleeper
        self.rng = rng or random.Random()
        self.breaker = breaker or CircuitBreaker(
            max_failures=config.circuit_max_failures,
            cooldown_seconds=config.circuit_cooldown_seconds,
        )
        self.last_retries: list[RetryEvent] = []

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.breaker.check()
        self.last_retries = []
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries):
            try:
                request = urllib.request.Request(
                    self.config.server_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(
                    request, timeout=self.config.request_timeout_seconds
                ) as response:
                    body = response.read().decode("utf-8")
                parsed = json.loads(body)
                if not isinstance(parsed, dict):
                    raise TypeError("mock server response must be an object")
                self.breaker.record_success()
                return parsed
            except urllib.error.HTTPError as exc:
                last_error = exc
                retry_after = exc.headers.get("Retry-After")
                if exc.code not in (429, 529) or attempt == self.config.max_retries - 1:
                    self.breaker.record_failure()
                    raise NetworkFailureError(f"HTTP {exc.code}: {exc.reason}") from exc
                delay = self._retry_delay(attempt, retry_after)
                self._record_retry(attempt, f"HTTP {exc.code}", delay)
            except (
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
                ValueError,
                TypeError,
            ) as exc:
                last_error = exc
                if attempt == self.config.max_retries - 1:
                    self.breaker.record_failure()
                    raise NetworkFailureError(f"mock server failed: {exc}") from exc
                delay = self._retry_delay(attempt)
                self._record_retry(attempt, type(exc).__name__, delay)

        self.breaker.record_failure()
        raise NetworkFailureError(f"mock server failed: {last_error}")

    def _retry_delay(self, attempt: int, retry_after: str | None = None) -> float:
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
        cap = self.config.retry_base_delay * (2**attempt)
        return cap + self.rng.uniform(0.0, self.config.retry_base_delay)

    def _record_retry(self, attempt: int, reason: str, delay: float) -> None:
        self.last_retries.append(RetryEvent(attempt=attempt + 1, reason=reason, delay_seconds=delay))
        self.sleeper(delay)


class LLMCaller(Protocol):
    last_retries: list[Any]

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class ValidatedAgentResponse:
    assistant_text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    partial_turn: bool = False
    expected_tool_call_count: int | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)
    message_count: int = 0
    token_count: int = 0


class AgentValidator:
    """Builds a model turn, calls mockllm, and records parsed response details."""

    def __init__(
        self,
        db: AgentDatabase,
        tracer: TraceWriter,
        llm_client: LLMCaller,
    ) -> None:
        self.db = db
        self.tracer = tracer
        self.llm_client = llm_client

    def validate_turn(
        self,
        run_id: str,
        *,
        task: str,
        step: int,
        memory: MemoryManager,
    ) -> ValidatedAgentResponse:
        context = memory.get_compacted_context()
        token_count = memory.estimated_tokens()
        llm_request_payload = {"message_count": len(context), "tokens": token_count}
        request_inserted = self.db.append_event(
            run_id,
            "llm_request",
            llm_request_payload,
            step=step,
            idempotency_key=f"{run_id}:llm_request:{step}",
        )
        if request_inserted is not None:
            self.tracer.log_trace(run_id, "llm_request", llm_request_payload, step=step)

        response = self.llm_client.call({"messages": context, "task": task, "step": step})
        for retry in getattr(self.llm_client, "last_retries", []):
            self.tracer.log_trace(run_id, "retry", _retry_payload(retry), step=step)
        response_inserted = self.db.append_event(
            run_id,
            "llm_response",
            response,
            step=step,
            idempotency_key=f"{run_id}:llm_response:{step}",
        )
        if response_inserted is not None:
            self.tracer.log_trace(run_id, "llm_response", response, step=step)

        return self._record_parsed_response(
            run_id,
            step=step,
            memory=memory,
            response=response,
            message_count=len(context),
            token_count=token_count,
        )

    def validate_recorded_response(
        self,
        run_id: str,
        *,
        step: int,
        memory: MemoryManager,
        response: dict[str, Any],
        message_count: int,
        token_count: int,
    ) -> ValidatedAgentResponse:
        return self._record_parsed_response(
            run_id,
            step=step,
            memory=memory,
            response=response,
            message_count=message_count,
            token_count=token_count,
        )

    def _record_parsed_response(
        self,
        run_id: str,
        *,
        step: int,
        memory: MemoryManager,
        response: dict[str, Any],
        message_count: int,
        token_count: int,
    ) -> ValidatedAgentResponse:
        parsed_response = parse_agent_response(response, step)
        if parsed_response.assistant_text:
            payload = {"content": parsed_response.assistant_text}
            inserted = self.db.append_event(
                run_id,
                "assistant_message",
                payload,
                step=step,
                idempotency_key=_event_key(run_id, "assistant_message", step, payload),
            )
            if inserted is not None:
                memory.add_assistant_message(parsed_response.assistant_text)
                self.tracer.log_trace(run_id, "assistant_message", payload, step=step)

        for index, error in enumerate(parsed_response.parse_errors):
            payload = {"error": error}
            inserted = self.db.append_event(
                run_id,
                "tool_parse_error",
                payload,
                step=step,
                idempotency_key=_event_key(
                    run_id,
                    f"tool_parse_error:{index}",
                    step,
                    payload,
                ),
            )
            if inserted is not None:
                self.tracer.log_trace(run_id, "tool_parse_error", payload, step=step)

        return ValidatedAgentResponse(
            assistant_text=parsed_response.assistant_text,
            tool_calls=parsed_response.tool_calls,
            parse_errors=parsed_response.parse_errors,
            partial_turn=parsed_response.partial_turn,
            expected_tool_call_count=parsed_response.expected_tool_call_count,
            raw_response=response,
            message_count=message_count,
            token_count=token_count,
        )


def _retry_payload(retry: Any) -> dict[str, Any]:
    if is_dataclass(retry) and not isinstance(retry, type):
        return asdict(retry)
    if isinstance(retry, dict):
        return retry
    payload = {
        key: getattr(retry, key)
        for key in ("attempt", "reason", "delay_seconds")
        if hasattr(retry, key)
    }
    return payload or {"retry": str(retry)}


def _event_key(run_id: str, event_type: str, step: int, payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()
    return f"{run_id}:{event_type}:{step}:{digest}"
