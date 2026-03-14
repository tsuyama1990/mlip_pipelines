import shutil
from pathlib import Path

import pytest

from src.dynamics.security_utils import validate_executable_path


def test_valid_trusted_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_bin = tmp_path / "lmp"
    dummy_bin.touch()
    dummy_bin.chmod(0o755)

    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: str(dummy_bin))

    res = validate_executable_path("lmp", [str(tmp_path)])
    assert res == str(dummy_bin.resolve(strict=True))


def test_symlink_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    link_path = tmp_path / "lmp_link"
    target_path = tmp_path / "lmp"
    target_path.touch()
    target_path.chmod(0o755)
    link_path.symlink_to(target_path)

    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: str(link_path))

    with pytest.raises(ValueError, match="Binary cannot be a symlink"):
        validate_executable_path("lmp", [str(tmp_path)])


def test_untrusted_path_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_bin = tmp_path / "lmp"
    dummy_bin.touch()
    dummy_bin.chmod(0o755)

    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: str(dummy_bin))

    with pytest.raises(ValueError, match="Resolved binary must reside in a trusted directory"):
        # tmp_path is not in the empty trusted directories list
        validate_executable_path("lmp", [])


def test_missing_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="Executable not found"):
        validate_executable_path("lmp", [])
