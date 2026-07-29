"""Structured errors for the custom agent runtime."""


class AgentError(Exception):
    """Base class for runtime errors surfaced to the CLI and traces."""


class AgentBaseError(AgentError):
    """Compatibility alias used by older loop code."""


class AgentConfigError(AgentError):
    """Raised when runtime configuration cannot be loaded or validated."""


class NetworkFailureError(AgentError):
    """Raised when the mock LLM server cannot be reached or returns bad data."""


class NetworkError(NetworkFailureError):
    """Compatibility alias for network failures."""


class CircuitBreakerOpenError(NetworkFailureError):
    """Raised when the circuit breaker refuses network calls."""


class ContextLimitExceededError(AgentError):
    """Raised when the 8,000-token context ceiling cannot be preserved."""


class BudgetExceededError(AgentError):
    """Raised when the simulated cumulative run budget is exhausted."""


class ModelContradictionError(AgentError):
    """Raised when the assistant claims success for a failed tool operation."""


class PartialToolTurnError(AgentError):
    """Raised when the model emits an incomplete interrupted tool-call turn."""


class MemoryLimitError(ContextLimitExceededError):
    """Compatibility alias for context budget failures."""


class SecurityViolationError(AgentError):
    """Raised when a tool request violates the runtime security boundary."""


class MaxStepsReachedError(AgentError):
    """Raised when the agent reaches the configured step ceiling."""


class NoProgressError(AgentError):
    """Raised when the agent repeats the same tool call without progress."""


class ToolArgumentError(AgentError):
    """Raised when a model emits malformed or wrong-typed tool arguments."""


class ToolExecutionError(AgentError):
    """Raised when a tool fails in a controlled, model-legible way."""


class ReplayError(AgentError):
    """Raised when a recorded run cannot be replayed."""
