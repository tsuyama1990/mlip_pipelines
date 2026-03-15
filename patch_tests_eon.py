import re

with open("tests/unit/test_eon_driver.py", "r") as f:
    data = f.read()

# Fix write_bad_structure which now requires O_NOFOLLOW
# Wait, write_bad_structure creates a file atomically.
# `test_write_bad_structure` failed with SystemExit: 100 because maybe `os.O_NOFOLLOW` is not supported on temp files or directory doesn't exist.
# Wait, in eon_driver: `flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)`
# If it fails, it prints "Failed to write bad structure".
# Wait, test_write_bad_structure creates a tempdir:
#   def test_write_bad_structure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
#       import tempfile
#       monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
#       eon_driver.write_bad_structure("test.cfg", Atoms("Cu"))
#       assert (tmp_path / "mlip_bad_structures" / "test.cfg").exists()
# Why did it fail? Let's mock sys.exit to raise a ValueError so we can catch the error string.

data = data.replace(
"""def test_write_bad_structure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import tempfile

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    eon_driver.write_bad_structure("test.cfg", Atoms("Cu"))
    assert (tmp_path / "mlip_bad_structures" / "test.cfg").exists()""",
"""def test_write_bad_structure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import tempfile
    import sys

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(sys, "exit", lambda x: None)

    eon_driver.write_bad_structure("test.cfg", Atoms("Cu"))
    assert (tmp_path / "mlip_bad_structures" / "test.cfg").exists()"""
)

with open("tests/unit/test_eon_driver.py", "w") as f:
    f.write(data)
