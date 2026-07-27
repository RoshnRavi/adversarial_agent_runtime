"""Custom errors for the agent runtime."""

class AgentError(Exception):
    """The main error class for all agent errors."""
    pass

class NetworkFailureError(AgentError):
    """Raised when the mockllm server connection fails."""
    pass

class CircuitBreakerOpenError(AgentError):
    """Raised when the Circuit Breaker stops network calls."""
    pass

class ContextLimitExceededError(AgentError):
    """Raised when the 8,000 token limit is reached."""
    pass

class SecurityViolationError(AgentError):
    """Raised when the AI tries a prompt injection or bad file path."""
    pass

class MaxStepsReachedError(AgentError):
    """Raised when the agent gets stuck in an infinite loop."""
    pass