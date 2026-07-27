import time
import random
import urllib.request
import urllib.error
import json
from dataclasses import dataclass

from .database import AgentDatabase, RunStateDTO
from .exceptions import (
    NetworkFailureError, 
    CircuitBreakerOpenError, 
    MaxStepsReachedError
)

class CircuitBreaker:
    """Stops network calls if the server fails too many times."""
    def __init__(self, max_failures: int = 5, cooldown: int = 60):
        self.max_failures = max_failures
        self.cooldown = cooldown
        self.failure_count = 0
        self.last_failure_time = 0.0

    def check(self):
        """Throws an error if the breaker is open."""
        if self.failure_count >= self.max_failures:
            if time.time() - self.last_failure_time < self.cooldown:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN. Server is dead.")
            else:
                # Cooldown finished, try again
                self.failure_count = 0

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()

    def record_success(self):
        self.failure_count = 0

class AgentLoop:
    """The main Finite State Machine loop for the agent."""
    
    # Dependency Injection: The database is passed in
    def __init__(self, db: AgentDatabase, server_url: str = "http://localhost:8000"):
        self.db = db
        self.server_url = server_url
        self.breaker = CircuitBreaker()
        self.max_steps = 30

    def _call_server_with_retry(self, payload: dict) -> dict:
        """Calls the mockllm server using Exponential Backoff and Jitter."""
        self.breaker.check()
        
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                data = json.dumps(payload).encode('utf-8')
                req = urllib.request.Request(
                    self.server_url, 
                    data=data, 
                    headers={'Content-Type': 'application/json'}
                )
                
                with urllib.request.urlopen(req, timeout=5) as response:
                    result = json.loads(response.read().decode('utf-8'))
                    self.breaker.record_success()
                    return result

            except (urllib.error.URLError, json.JSONDecodeError) as e:
                if attempt == max_retries - 1:
                    self.breaker.record_failure()
                    raise NetworkFailureError(f"Server failed after {max_retries} tries: {e}")
                
                # Exponential Backoff with Jitter
                sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0, 0.5)
                time.sleep(sleep_time)

    def run_task(self, run_id: str, task: str):
        """Runs the agent loop step-by-step."""
        state = RunStateDTO(run_id=run_id, current_state="START", step_count=0)
        self.db.save_state(state)

        while state.current_state != "FINISHED":
            if state.step_count >= self.max_steps:
                raise MaxStepsReachedError("Agent is stuck in an infinite loop.")

            state.current_state = "CALLING_LLM"
            self.db.save_state(state)
            
            try:
                # Ask the AI what to do
                response = self._call_server_with_retry({"task": task, "step": state.step_count})
                
                # Simulated tool logic would go here
                # ...
                
                state.step_count += 1
                if state.step_count > 5: # Just a simulation to finish
                    state.current_state = "FINISHED"
                
                self.db.save_state(state)

            except NetworkFailureError as e:
                print(f"Network error: {e}")
                break