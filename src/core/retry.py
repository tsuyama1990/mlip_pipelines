import enum
import logging
import threading
import time
from collections.abc import Callable
from typing import Any


class CircuitBreakerState(enum.Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 300.0) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitBreakerState.CLOSED
        self.failures = 0
        self.last_failure_time = 0.0
        self._lock = threading.Lock()

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logging.error("Circuit breaker tripped to OPEN.")

    def record_success(self) -> None:
        with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                logging.info("Circuit breaker recovered, transitioning to CLOSED.")
            self.state = CircuitBreakerState.CLOSED
            self.failures = 0

    def can_execute(self) -> bool:
        with self._lock:
            if self.state == CircuitBreakerState.CLOSED:
                return True
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    logging.info("Circuit breaker transitioning to HALF_OPEN to test recovery.")
                    return True
                return False
            # state == HALF_OPEN: only allow 1 test execution
            return True


class RetryManager:
    """Manages retry logic and circuit breaking for arbitrary operations."""

    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        timeout: float = 30.0,
        circuit_breaker_cooldown: float = 300.0,
    ) -> None:
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=max_retries, recovery_timeout=circuit_breaker_cooldown
        )

    def execute_with_retry(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        import concurrent.futures

        if not self.circuit_breaker.can_execute():
            msg = "Circuit breaker is OPEN. Operation aborted."
            raise RuntimeError(msg)

        for attempt in range(self.max_retries):
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(operation, *args, **kwargs)
                    result = future.result(timeout=self.timeout)
            except concurrent.futures.TimeoutError as e:
                logging.warning(f"Operation timed out after {self.timeout} seconds.")
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_factor**attempt)
                else:
                    self.circuit_breaker.record_failure()
                    msg = "Transient timeout failures exhausted."
                    raise RuntimeError(msg) from e
            except (ConnectionError, TimeoutError, OSError) as e:
                # Known transient exceptions
                if attempt < self.max_retries - 1:
                    sleep_time = self.backoff_factor**attempt
                    logging.warning(f"Transient failure: {e}. Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    self.circuit_breaker.record_failure()
                    msg = "Transient IO failures exhausted."
                    raise RuntimeError(msg) from e
            except Exception:
                # Permanent configuration or infrastructure issues
                self.circuit_breaker.record_failure()
                raise
            else:
                self.circuit_breaker.record_success()
                return result

        unreachable_msg = "Unreachable"
        raise RuntimeError(unreachable_msg)
