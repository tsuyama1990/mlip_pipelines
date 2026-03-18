import logging
import time
from collections.abc import Callable
from typing import Any


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
        self.circuit_breaker_cooldown = circuit_breaker_cooldown

        self._circuit_breaker_open = False
        self._circuit_breaker_reset_time = 0.0

    def check_circuit_breaker(self) -> bool:
        """Returns True if the circuit breaker is OPEN and operation should fail fast."""
        if self._circuit_breaker_open:
            if time.time() < self._circuit_breaker_reset_time:
                return True
            self._circuit_breaker_open = False
        return False

    def execute_with_retry(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        import concurrent.futures

        if self.check_circuit_breaker():
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
                    self._trip_circuit_breaker()
                    msg = "Transient timeout failures exhausted."
                    raise RuntimeError(msg) from e
            except (ConnectionError, TimeoutError, OSError) as e:
                # Known transient exceptions
                if attempt < self.max_retries - 1:
                    sleep_time = self.backoff_factor**attempt
                    logging.warning(f"Transient failure: {e}. Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    self._trip_circuit_breaker()
                    msg = "Transient IO failures exhausted."
                    raise RuntimeError(msg) from e
            except Exception:
                # Permanent configuration or infrastructure issues
                raise
            else:
                # Success closes the breaker if it was half-open
                self._circuit_breaker_open = False
                return result

        unreachable_msg = "Unreachable"
        raise RuntimeError(unreachable_msg)

    def _trip_circuit_breaker(self) -> None:
        self._circuit_breaker_open = True
        self._circuit_breaker_reset_time = time.time() + self.circuit_breaker_cooldown
        logging.exception("Tripping circuit breaker for operation.")
