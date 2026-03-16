import json
import logging
import sqlite3
from pathlib import Path
from typing import Any


class CheckpointManager:
    """Manages the application state and artifacts using a SQLite database."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        try:
            with sqlite3.connect(self.db_path, isolation_level='IMMEDIATE') as conn:
                conn.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT)")
        except sqlite3.OperationalError as e:
            msg = f"Failed to initialize checkpoint database at {self.db_path}. It may be locked or corrupted. Please check file permissions or remove the lock."
            logging.exception(msg)
            raise RuntimeError(msg) from e

    def set_state(self, key: str, value: Any) -> None:
        """Sets a state value in the database with JSON serialization."""
        import time
        try:
            json_value = json.dumps(value)
        except TypeError as e:
            msg = f"Value for key '{key}' is not JSON serializable."
            logging.exception(msg)
            raise ValueError(msg) from e

        max_retries = 5
        base_delay = 0.1
        for attempt in range(max_retries):
            try:
                with sqlite3.connect(self.db_path, isolation_level=None, timeout=20.0) as conn:
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute("PRAGMA synchronous=NORMAL;")
                    with conn:
                        conn.execute(
                            "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
                            (key, json_value),
                        )
            except sqlite3.OperationalError as e:
                if attempt == max_retries - 1:
                    msg = f"Failed to write state '{key}' to database. Database may be locked."
                    logging.exception(msg)
                    raise RuntimeError(msg) from e
                time.sleep(base_delay * (2 ** attempt))

    def get_state(self, key: str) -> Any | None:
        """Gets a state value from the database, deserializing from JSON."""
        import time
        max_retries = 5
        base_delay = 0.1
        for attempt in range(max_retries):
            try:
                with sqlite3.connect(self.db_path, isolation_level=None, timeout=20.0) as conn:
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute("PRAGMA synchronous=NORMAL;")
                    cursor = conn.execute("SELECT value FROM state WHERE key = ?", (key,))
                    row = cursor.fetchone()
                    if row is None:
                        return None
                    return json.loads(row[0])
            except sqlite3.OperationalError as e:
                if attempt == max_retries - 1:
                    msg = f"Failed to read state '{key}' from database. Database may be locked."
                    logging.exception(msg)
                    raise RuntimeError(msg) from e
                time.sleep(base_delay * (2 ** attempt))
            except json.JSONDecodeError as e:
                msg = f"Failed to decode JSON value for key '{key}'."
                logging.exception(msg)
                raise ValueError(msg) from e
        return None
