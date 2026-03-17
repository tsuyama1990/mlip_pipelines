import sqlite3
from pathlib import Path

import pytest

from src.core.checkpoint import CheckpointManager


def test_checkpoint_manager_init(tmp_path: Path):
    db_path = tmp_path / "checkpoint.db"
    CheckpointManager(db_path)
    assert db_path.exists()

def test_checkpoint_set_and_get(tmp_path: Path):
    db_path = tmp_path / "checkpoint.db"
    cm = CheckpointManager(db_path)

    cm.set_state("CURRENT_PHASE", "PHASE1")
    assert cm.get_state("CURRENT_PHASE") == "PHASE1"

    # Complex dict
    complex_val = {"iteration": 5, "halted": True, "indices": [1, 2, 3]}
    cm.set_state("HALT_INFO", complex_val)
    assert cm.get_state("HALT_INFO") == complex_val

def test_checkpoint_transaction_rollback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "checkpoint.db"
    cm = CheckpointManager(db_path)
    cm.set_state("KEY", "SAFE_VALUE")

    # Force a failure during set_state
    op_err_msg = "Forced Failure"

    def mock_execute(*args, **kwargs):
        raise sqlite3.OperationalError(op_err_msg)

    original_connect = sqlite3.connect
    class MockConnection:
        def __init__(self, *args, **kwargs) -> None:
            self.conn = original_connect(*args, **kwargs)
        def __enter__(self) -> "MockConnection":
            return self
        def __exit__(self, exc_type, exc_val, exc_tb) -> None:
            self.conn.close()
        def execute(self, *args, **kwargs) -> None:
            raise sqlite3.OperationalError(op_err_msg)

    monkeypatch.setattr(sqlite3, "connect", MockConnection)

    with pytest.raises(RuntimeError, match="Failed to set state KEY in checkpoint database"):
        cm.set_state("KEY", "DANGEROUS_VALUE")

    monkeypatch.undo()
    # It should not have been overwritten
    assert cm.get_state("KEY") == "SAFE_VALUE"


def test_checkpoint_invalid_db_path(tmp_path: Path):
    with pytest.raises(ValueError, match="Directory cannot be a system directory|must reside securely within an allowed base directory"):
        CheckpointManager(Path("/etc/shadow_db.sqlite"))

def test_checkpoint_get_error(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "checkpoint.db"
    cm = CheckpointManager(db_path)

    original_connect = sqlite3.connect
    class MockConnectionGet:
        def __init__(self, *args, **kwargs):
            self.conn = original_connect(*args, **kwargs)
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.conn.close()
        def execute(self, *args, **kwargs):
            raise sqlite3.OperationalError("Forced Get Failure")

    monkeypatch.setattr(sqlite3, "connect", MockConnectionGet)
    with pytest.raises(RuntimeError, match="Failed to get state MISSING from checkpoint database"):
        cm.get_state("MISSING")
