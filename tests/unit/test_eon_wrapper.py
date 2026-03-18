# ruff: noqa: S108
from pathlib import Path
from typing import Any

import pytest

from src.domain_models.config import DynamicsConfig, SystemConfig
from src.dynamics.eon_wrapper import EONWrapper


@pytest.fixture
def config(tmp_path: Path) -> DynamicsConfig:
    return DynamicsConfig(
        trusted_directories=[],
        project_root=str(tmp_path),
        uncertainty_threshold=5.0,
    )


@pytest.fixture
def sys_config() -> SystemConfig:
    return SystemConfig(elements=["Fe", "Pt"], baseline_potential="zbl")


def test_eon_wrapper_ini_creation(
    tmp_path: Path, config: DynamicsConfig, sys_config: SystemConfig
) -> None:
    wrapper = EONWrapper(config, sys_config)
    wrapper._write_config_ini(tmp_path)
    ini_file = tmp_path / "config.ini"
    assert ini_file.exists()
    content = ini_file.read_text()
    assert "[Process Search]" in content
    assert "min_mode_method = dimer" in content
    assert "job = process_search" in content


def test_eon_wrapper_driver_creation(
    tmp_path: Path, config: DynamicsConfig, sys_config: SystemConfig
) -> None:
    wrapper = EONWrapper(config, sys_config)
    potential = tmp_path / "potential.yace"
    potential.touch()
    wrapper._write_pace_driver(tmp_path, potential)
    driver_file = tmp_path / "potentials" / "pace_driver.py"
    assert driver_file.exists()

    # Must be executable
    import os

    assert os.access(driver_file, os.X_OK)


def test_eon_wrapper_run_kmc_no_halt(
    tmp_path: Path,
    config: DynamicsConfig,
    sys_config: SystemConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # We mock shutil.which so it acts like eonclient is not found, letting the FileNotFoundError block handle it
    import shutil

    def mock_which(name: str) -> str | None:
        return None

    monkeypatch.setattr(shutil, "which", mock_which)

    wrapper = EONWrapper(config, sys_config)
    with pytest.raises(RuntimeError, match="EON client executable not found"):
        wrapper.run_kmc(None, tmp_path)


def test_write_config_ini_invalid_eon_job(tmp_path: Path) -> None:
    config = DynamicsConfig(trusted_directories=[], project_root=str(tmp_path))
    config.eon_job = "job;"
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = EONWrapper(config, sys_config)
    with pytest.raises(ValueError, match="Invalid characters in filename:"):
        engine._write_config_ini(tmp_path)


def test_write_config_ini_invalid_min_mode_method(tmp_path: Path) -> None:
    config = DynamicsConfig(trusted_directories=[], project_root=str(tmp_path))
    config.eon_min_mode_method = "method;"
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = EONWrapper(config, sys_config)
    with pytest.raises(ValueError, match="Invalid characters in filename:"):
        engine._write_config_ini(tmp_path)


def test_write_pace_driver_invalid_potential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "shutil.which", lambda x: "/usr/bin/eonclient" if x == "eonclient" else None
    )
    config = DynamicsConfig(
        project_root=str(tmp_path),
        eon_binary="eonclient",
        trusted_directories=[],
        eon_job="dummy",
        eon_min_mode_method="dummy",
    )
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = EONWrapper(config, sys_config)

    # Potential path is outside project root
    pot_path = tmp_path.parent / "hacker_dummy.yace"
    pot_path.parent.mkdir(parents=True, exist_ok=True)
    pot_path.touch()

    with pytest.raises(ValueError, match="Potential path must be within the project root"):
        engine._write_pace_driver(tmp_path, pot_path)


def test_write_pace_driver_invalid_python_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = DynamicsConfig(trusted_directories=[], project_root=str(tmp_path))
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = EONWrapper(config, sys_config)

    import sys

    monkeypatch.setattr(sys, "executable", "python;")

    with pytest.raises(ValueError, match="Invalid python executable path"):
        engine._write_pace_driver(tmp_path, None)


def test_run_kmc_invalid_work_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "shutil.which", lambda x: "/usr/bin/eonclient" if x == "eonclient" else None
    )
    config = DynamicsConfig(
        project_root=str(tmp_path),
        eon_binary="eonclient",
        trusted_directories=[],
        eon_job="dummy",
        eon_min_mode_method="dummy",
    )
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = EONWrapper(config, sys_config)

    work_dir = Path("/var/tmp/hacker_work")
    with pytest.raises(ValueError, match="Path outside allowed directories"):
        engine.run_kmc(None, work_dir)


def test_run_kmc_subprocess_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import shutil

    mock_bin_dir = tmp_path / "bin"
    mock_bin_dir.mkdir(parents=True, exist_ok=True)
    mock_eon = mock_bin_dir / "eonclient"
    mock_eon.touch()
    mock_eon.chmod(0o755)

    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: str(mock_eon.resolve()))

    config = DynamicsConfig.model_construct(
        project_root=str(tmp_path),
        eon_binary="eonclient",
        trusted_directories=[str(mock_bin_dir)],
        eon_job="dummy",
        eon_min_mode_method="dummy",
    )
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = EONWrapper(config, sys_config)

    class MockProc:
        returncode = 1

        def __init__(self, cmd: list[str]) -> None:
            if not cmd or "eonclient" not in cmd[0]:
                msg = "Invalid EON command"
                raise ValueError(msg)

        def communicate(self, *args: Any, **kwargs: Any) -> tuple[bytes, bytes]:
            return b"out", b"err"

        def kill(self) -> None:
            return None

        def poll(self) -> int | None:
            return self.returncode

        def __enter__(self) -> "MockProc":
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            return None

    import subprocess

    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: MockProc(cmd))

    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match="EON client failed with return code"):
        engine.run_kmc(None, work_dir)


def test_run_kmc_subprocess_halted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import shutil

    mock_bin_dir = tmp_path / "bin"
    mock_bin_dir.mkdir(parents=True, exist_ok=True)
    mock_eon = mock_bin_dir / "eonclient"
    mock_eon.touch()
    mock_eon.chmod(0o755)

    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: str(mock_eon.resolve()))

    config = DynamicsConfig.model_construct(
        project_root=str(tmp_path),
        eon_binary="eonclient",
        trusted_directories=[str(mock_bin_dir)],
        eon_job="dummy",
        eon_min_mode_method="dummy",
    )
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = EONWrapper(config, sys_config)

    class MockProc:
        returncode = 100

        def __init__(self, cmd: list[str]) -> None:
            if not cmd or "eonclient" not in cmd[0]:
                msg = "Invalid EON command"
                raise ValueError(msg)

        def communicate(self, *args: Any, **kwargs: Any) -> tuple[bytes, bytes]:
            return b"out", b"err"

        def kill(self) -> None:
            return None

        def poll(self) -> int | None:
            return self.returncode

        def __enter__(self) -> "MockProc":
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            return None

    import subprocess

    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: MockProc(cmd))

    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    res = engine.run_kmc(None, work_dir)
    assert res["halted"] is True
    assert res["is_kmc"] is True


def test_run_kmc_missing_executable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "shutil.which", lambda x: "/usr/bin/eonclient" if x == "eonclient" else None
    )
    config = DynamicsConfig(
        project_root=str(tmp_path),
        eon_binary="eonclient",
        trusted_directories=[],
        eon_job="dummy",
        eon_min_mode_method="dummy",
    )
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = EONWrapper(config, sys_config)

    from typing import Any

    import src.dynamics.security_utils

    def mock_val(*args: Any, **kwargs: Any) -> str:
        msg = "EON client executable not found."
        raise RuntimeError(msg)

    monkeypatch.setattr(src.dynamics.security_utils, "validate_executable_path", mock_val)

    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match="EON client executable not found."):
        engine.run_kmc(None, work_dir)


def test_validate_work_dir_outside_root(tmp_path: Path):
    config = DynamicsConfig(project_root=str(tmp_path), trusted_directories=[])
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = EONWrapper(config, sys_config)

    with pytest.raises(ValueError, match="Path traversal sequences"):
        engine._validate_work_dir(tmp_path / "../outside")


def test_get_validated_eon_bin_not_executable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dummy_bin_dir = tmp_path / "bin"
    dummy_bin_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "shutil.which", lambda x: "/usr/bin/eonclient" if x == "eonclient" else None
    )

    config = DynamicsConfig(
        project_root=str(tmp_path),
        eon_binary="eonclient",
        trusted_directories=[str(dummy_bin_dir)],
        eon_job="dummy",
        eon_min_mode_method="dummy",
    )
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = EONWrapper(config, sys_config)

    dummy = dummy_bin_dir / "eonclient"
    dummy.touch()
    dummy.chmod(0o644)  # Not executable

    import shutil

    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: str(dummy))

    with pytest.raises(ValueError, match="is not an executable file"):
        engine._get_validated_eon_bin()


def test_build_safe_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = DynamicsConfig(
        project_root=str(tmp_path), trusted_directories=["/usr/bin", "/bin", "/usr/lib"]
    )
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = EONWrapper(config, sys_config)

    import os

    monkeypatch.setenv("PATH", os.pathsep.join(["/usr/bin", "/bin"]))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/lib")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")

    env = engine._build_safe_env()
    assert env["PATH"] == os.pathsep.join(
        [Path("/usr/bin").resolve().as_posix(), Path("/bin").resolve().as_posix()]
    )
    assert env["LD_LIBRARY_PATH"] == Path("/usr/lib").resolve().as_posix()
    assert env["OMP_NUM_THREADS"] == "4"


def test_run_exploration_eon(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import shutil
    import subprocess

    mock_bin_dir = tmp_path / "bin"
    mock_bin_dir.mkdir(parents=True, exist_ok=True)
    mock_eon = mock_bin_dir / "eonclient"
    mock_eon.touch()
    mock_eon.chmod(0o755)

    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: str(mock_eon.resolve()))

    config = DynamicsConfig.model_construct(
        project_root=str(tmp_path),
        eon_binary="eonclient",
        trusted_directories=[str(mock_bin_dir)],
        eon_job="dummy",
        eon_min_mode_method="dummy",
    )
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = EONWrapper(config, sys_config)

    class MockProc:
        returncode = 0

        def __init__(self, cmd: list[str]) -> None:
            if not cmd or "eonclient" not in cmd[0]:
                msg = "Invalid EON command"
                raise ValueError(msg)

        def communicate(self, *args: Any, **kwargs: Any) -> tuple[bytes, bytes]:
            return b"out", b"err"

        def kill(self) -> None:
            return None

        def poll(self) -> int | None:
            return self.returncode

        def __enter__(self) -> "MockProc":
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            return None

    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: MockProc(cmd))

    work_dir = tmp_path / "work"
    res = engine.run_exploration(None, work_dir)
    assert res["halted"] is False


def test_run_kmc_untrusted_binary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import shutil

    mock_bin_dir = tmp_path / "bin"
    mock_bin_dir.mkdir(parents=True, exist_ok=True)
    mock_eon = mock_bin_dir / "eonclient"
    mock_eon.touch()
    mock_eon.chmod(0o755)

    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: str(mock_eon.resolve()))

    # Needs to exist to resolve it successfully
    other_trusted = tmp_path / "other_trusted"
    other_trusted.mkdir(parents=True, exist_ok=True)

    config = DynamicsConfig.model_construct(
        project_root=str(tmp_path),
        eon_binary="eonclient",
        trusted_directories=[str(other_trusted)],
        eon_job="dummy",
        eon_min_mode_method="dummy",
    )
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = EONWrapper(config, sys_config)

    # Note: validation will find eonclient via which, but its directory won't be in the trusted list

    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="EON binary is not within trusted directories."):
        engine.run_kmc(None, work_dir)


def test_run_kmc_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import shutil
    import subprocess

    mock_bin_dir = tmp_path / "bin"
    mock_bin_dir.mkdir(parents=True, exist_ok=True)
    mock_eon = mock_bin_dir / "eonclient"
    mock_eon.touch()
    mock_eon.chmod(0o755)

    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: str(mock_eon.resolve()))

    config = DynamicsConfig.model_construct(
        project_root=str(tmp_path),
        eon_binary="eonclient",
        trusted_directories=[str(mock_bin_dir)],
        eon_job="dummy",
        eon_min_mode_method="dummy",
    )
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = EONWrapper(config, sys_config)

    class MockProc:
        returncode = 0

        def __init__(self, cmd: list[str]) -> None:
            self.count = 0
            if not cmd or "eonclient" not in cmd[0]:
                msg = "Invalid EON command"
                raise ValueError(msg)

        def poll(self) -> int | None:
            return self.returncode

        def communicate(self, *args: Any, **kwargs: Any) -> tuple[bytes, bytes]:
            if self.count == 0:
                self.count += 1
                raise subprocess.TimeoutExpired(cmd="eonclient", timeout=3600)
            return b"out", b"err"

        def kill(self) -> None:
            return None

        def __enter__(self) -> "MockProc":
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            return None

    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: MockProc(cmd))

    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(RuntimeError, match="EON client execution timed out"):
        engine.run_exploration(None, work_dir)
