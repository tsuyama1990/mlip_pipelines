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
    pot_path = Path("/var/tmp/hacker_dummy.yace")
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
    with pytest.raises(ValueError, match="must reside securely within an allowed base directory"):
        engine.run_kmc(None, work_dir)


def test_run_kmc_subprocess_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    dummy.chmod(0o755)

    import shutil

    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: str(dummy))

    class MockProc:
        returncode = 1

        def communicate(self, *args, **kwargs):
            return b"out", b"err"

        def kill(self):
            pass

        def poll(self) -> int | None:
            return self.returncode

        def __enter__(self) -> "MockProc":
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

    import subprocess

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: MockProc())

    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match="EON client failed with return code"):
        engine.run_kmc(None, work_dir)


def test_run_kmc_subprocess_halted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    dummy.chmod(0o755)

    import shutil

    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: str(dummy))

    class MockProc:
        returncode = 100

        def communicate(self, *args, **kwargs):
            return b"out", b"err"

        def kill(self):
            pass

        def poll(self) -> int | None:
            return self.returncode

        def __enter__(self) -> "MockProc":
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

    import subprocess

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: MockProc())

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

    with pytest.raises(ValueError, match="is outside the allowed project root"):
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


def test_run_exploration_eon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import subprocess

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

    dummy_bin = dummy_bin_dir / "eonclient"
    dummy_bin.touch()
    dummy_bin.chmod(0o755)
    import shutil

    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: str(dummy_bin))

    class MockProc:
        returncode = 0

        def communicate(self, *args, **kwargs):
            return b"out", b"err"

        def kill(self):
            pass

        def poll(self) -> int | None:
            return self.returncode

        def __enter__(self) -> "MockProc":
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: MockProc())

    work_dir = tmp_path / "work"
    res = engine.run_exploration(None, work_dir)
    assert res["halted"] is False


def test_run_kmc_untrusted_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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

    # We mock validate_executable_path to return a valid path but then the test will fail in the eon wrapper because the path is not in trusted directory list
    import src.dynamics.security_utils

    dummy_bin = tmp_path / "eonclient"
    dummy_bin.touch()
    dummy_bin.chmod(0o755)
    import shutil

    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: str(dummy_bin))

    monkeypatch.setattr(
        src.dynamics.security_utils,
        "validate_executable_path",
        lambda *args, **kwargs: dummy_bin.resolve(strict=True),
    )
    work_dir = tmp_path / "work"

    with pytest.raises(ValueError, match="Resolved binary must reside in a trusted directory:"):
        engine.run_kmc(None, work_dir)


def test_run_kmc_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import subprocess

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

    dummy_bin = dummy_bin_dir / "eonclient"
    dummy_bin.touch()
    dummy_bin.chmod(0o755)
    import shutil

    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: str(dummy_bin))

    class MockProc:
        returncode = 0

        def __init__(self) -> None:
            self.count = 0

        def poll(self) -> int | None:
            return self.returncode

        def communicate(self, *args: Any, **kwargs: Any) -> tuple[bytes, bytes]:
            if self.count == 0:
                self.count += 1
                raise subprocess.TimeoutExpired(cmd="eonclient", timeout=3600)
            return b"out", b"err"

        def kill(self) -> None:
            pass

        def __enter__(self) -> "MockProc":
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: MockProc())

    work_dir = tmp_path / "work"
    with pytest.raises(RuntimeError, match="EON client execution timed out"):
        engine.run_exploration(None, work_dir)
