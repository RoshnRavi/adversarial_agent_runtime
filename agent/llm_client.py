"""Mock LLM HTTP client with retry, jitter, and circuit breaker support."""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from .config import DEFAULT_CONFIG, RuntimeConfig
from .exceptions import CircuitBreakerOpenError, NetworkFailureError


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
                    raise ValueError("mock server response must be an object")
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
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
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
