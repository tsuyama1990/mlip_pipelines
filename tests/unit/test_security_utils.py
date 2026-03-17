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
    assert str(res) == str(dummy_bin.resolve(strict=True))


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


import src.dynamics.security_utils as sec  # noqa: E402


def test_validate_env_key():
    sec._validate_env_key("MLIP_OK")
    with pytest.raises(ValueError, match="Only MLIP_ prefixes are allowed"):
        sec._validate_env_key("BAD")
    with pytest.raises(ValueError, match="Invalid characters"):
        sec._validate_env_key("MLIP_BAD;")


def test_validate_env_value():
    sec._validate_env_value("value")
    with pytest.raises(ValueError, match="Invalid characters"):
        sec._validate_env_value("val..ue")
    with pytest.raises(ValueError, match="Invalid characters"):
        sec._validate_env_value("val;ue")


def test_validate_env_file_security(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("MLIP_K=V")
    env_file.chmod(0o600)

    # Base directory mismatch
    with pytest.raises(ValueError, match="allowed base directory"):
        sec.validate_env_file_security(env_file, tmp_path / "other")

    # Success case
    assert sec.validate_env_file_security(env_file, tmp_path) == env_file.resolve()


def test_validate_and_copy_potential(tmp_path: Path):
    pot_dir = tmp_path / "potentials"
    pot_dir.mkdir()
    tmp_work = tmp_path / "work"
    tmp_work.mkdir()

    src_pot = tmp_work / "test.yace"
    src_pot.touch()

    res = sec.validate_and_copy_potential(src_pot, pot_dir, 1, tmp_work)
    assert res == pot_dir / "generation_001.yace"
    assert res.exists()


def test_validate_and_copy_potential_outside_work(tmp_path: Path):
    pot_dir = tmp_path / "potentials"
    pot_dir.mkdir()
    tmp_work = tmp_path / "work"
    tmp_work.mkdir()

    src_pot = tmp_path / "test.yace"  # Outside work
    src_pot.touch()

    with pytest.raises(ValueError, match="strictly reside within the tmp working directory"):
        sec.validate_and_copy_potential(src_pot, pot_dir, 1, tmp_work)
