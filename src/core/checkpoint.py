import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from src.domain_models.config import _secure_resolve_and_validate_dir

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages the active learning orchestrator's state to support HPC resuming."""

    def __init__(self, db_path: Path) -> None:
        """Initializes the checkpoint manager with a database path.

        Args:
            db_path: The absolute path to the SQLite database file.
        """
        # Validate the parent directory to prevent path traversal
        parent_dir = db_path.parent
        _secure_resolve_and_validate_dir(str(parent_dir), check_exists=True)

        self.db_path = db_path.resolve(strict=False)
        self._init_db()

    def _init_db(self) -> None:
        """Initializes the SQLite database with isolation_level="IMMEDIATE" for autocommit."""
        try:
            # We use IMMEDIATE to grab an exclusive lock early, but we still need autocommit
            # so we let the context manager handle the transaction.
            # Using isolation_level="IMMEDIATE" means autocommit mode.
            with sqlite3.connect(self.db_path, isolation_level="IMMEDIATE") as conn:
                # We can't use 'IMMEDIATE' natively with context manager if isolation_level="IMMEDIATE"
                # but we can execute standard queries safely.
                # If we want transactions, we manage them manually, or just rely on autocommit.
                conn.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT)")
        except sqlite3.OperationalError as e:
            msg = f"Failed to initialize checkpoint database at {self.db_path}: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e

    def set_state(self, key: str, value: Any) -> None:
        """Sets a state key-value pair in the database.

        Args:
            key: The state key.
            value: The state value, which will be JSON serialized.
        """
        try:
            serialized = json.dumps(value)
            with sqlite3.connect(self.db_path, isolation_level="IMMEDIATE") as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
                    (key, serialized),
                )
        except (sqlite3.OperationalError, TypeError) as e:
            msg = f"Failed to set state {key} in checkpoint database: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e

    def get_state(self, key: str) -> Any:
        """Retrieves a state value from the database.

        Args:
            key: The state key to retrieve.

        Returns:
            The JSON-deserialized value, or None if the key does not exist.
        """
        try:
            with sqlite3.connect(self.db_path, isolation_level="IMMEDIATE") as conn:
                cursor = conn.execute("SELECT value FROM state WHERE key = ?", (key,))
                row = cursor.fetchone()
                if row is None:
                    return None
                return json.loads(row[0])
        except (sqlite3.OperationalError, json.JSONDecodeError) as e:
            msg = f"Failed to get state {key} from checkpoint database: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e
