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
    # Set PATH so we explicitly do not find eonclient
    import os
    monkeypatch.setenv("PATH", str(tmp_path))

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
    dummy_eon = tmp_path / "eonclient"
    dummy_eon.write_text("#!/bin/bash\nexit 0")
    dummy_eon.chmod(0o755)

    import os
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ.get('PATH', '')}")
    config = DynamicsConfig(
        project_root=str(tmp_path),
        eon_binary="eonclient",
        trusted_directories=[str(tmp_path)],
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
    dummy_eon = tmp_path / "eonclient"
    dummy_eon.write_text("#!/bin/bash\nexit 0")
    dummy_eon.chmod(0o755)

    import os
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ.get('PATH', '')}")

    config = DynamicsConfig(
        project_root=str(tmp_path),
        eon_binary="eonclient",
        trusted_directories=[str(tmp_path)],
        eon_job="dummy",
        eon_min_mode_method="dummy",
    )
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = EONWrapper(config, sys_config)

    work_dir = Path("/var/tmp/hacker_work")
    with pytest.raises(ValueError, match="Path outside allowed directories"):
        engine.run_kmc(None, work_dir)


def test_run_kmc_subprocess_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_bin_dir = tmp_path / "bin"
    dummy_bin_dir.mkdir(parents=True, exist_ok=True)
    dummy_eon = dummy_bin_dir / "eonclient"
    dummy_eon.write_text("#!/bin/bash\nexit 1")
    dummy_eon.chmod(0o755)

    import os
    monkeypatch.setenv("PATH", f"{dummy_bin_dir}:{os.environ.get('PATH', '')}")

    config = DynamicsConfig(
        project_root=str(tmp_path),
        eon_binary="eonclient",
        trusted_directories=[str(dummy_bin_dir)],
        eon_job="dummy",
        eon_min_mode_method="dummy",
    )
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = EONWrapper(config, sys_config)

    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match="EON client failed with return code"):
        engine.run_kmc(None, work_dir)


def test_run_kmc_subprocess_halted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_bin_dir = tmp_path / "bin"
    dummy_bin_dir.mkdir(parents=True, exist_ok=True)
    dummy_eon = dummy_bin_dir / "eonclient"
    dummy_eon.write_text("#!/bin/bash\nexit 100")
    dummy_eon.chmod(0o755)

    import os
    monkeypatch.setenv("PATH", f"{dummy_bin_dir}:{os.environ.get('PATH', '')}")

    config = DynamicsConfig(
        project_root=str(tmp_path),
        eon_binary="eonclient",
        trusted_directories=[str(dummy_bin_dir)],
        eon_job="dummy",
        eon_min_mode_method="dummy",
    )
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = EONWrapper(config, sys_config)

    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    res = engine.run_kmc(None, work_dir)
    assert res["halted"] is True
    assert res["is_kmc"] is True


def test_run_kmc_missing_executable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os
    monkeypatch.setenv("PATH", str(tmp_path))

    # Create dummy config directly bypassing pydantic validation of PATH existence
    config = DynamicsConfig.model_construct(
        project_root=str(tmp_path),
        eon_binary="eonclient",
        trusted_directories=[],
        eon_job="dummy",
        eon_min_mode_method="dummy",
    )
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = EONWrapper(config, sys_config)

    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match="EON client executable not found"):
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
    dummy_eon = dummy_bin_dir / "eonclient"
    dummy_eon.touch()
    dummy_eon.chmod(0o644)  # Not executable

    import os
    monkeypatch.setenv("PATH", f"{dummy_bin_dir}:{os.environ.get('PATH', '')}")
    # Force shutil.which to find the non-executable file, bypassing its own internal access check
    import shutil
    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: str(dummy_eon))

    # Create dummy config directly bypassing pydantic validation of PATH existence
    config = DynamicsConfig.model_construct(
        project_root=str(tmp_path),
        eon_binary="eonclient",
        trusted_directories=[str(dummy_bin_dir)],
        eon_job="dummy",
        eon_min_mode_method="dummy",
    )
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = EONWrapper(config, sys_config)

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
    dummy_bin_dir = tmp_path / "bin"
    dummy_bin_dir.mkdir(parents=True, exist_ok=True)
    dummy_eon = dummy_bin_dir / "eonclient"
    dummy_eon.write_text("#!/bin/bash\nexit 0")
    dummy_eon.chmod(0o755)

    import os
    monkeypatch.setenv("PATH", f"{dummy_bin_dir}:{os.environ.get('PATH', '')}")

    config = DynamicsConfig(
        project_root=str(tmp_path),
        eon_binary="eonclient",
        trusted_directories=[str(dummy_bin_dir)],
        eon_job="dummy",
        eon_min_mode_method="dummy",
    )
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = EONWrapper(config, sys_config)

    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    res = engine.run_exploration(None, work_dir)
    assert res["halted"] is False


def test_run_kmc_untrusted_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dummy_bin_dir = tmp_path / "bin"
    dummy_bin_dir.mkdir(parents=True, exist_ok=True)

    # We place the binary OUTSIDE the trusted directory
    untrusted_dir = tmp_path / "untrusted"
    untrusted_dir.mkdir(parents=True, exist_ok=True)
    untrusted_eon = untrusted_dir / "eonclient"
    untrusted_eon.touch()
    untrusted_eon.chmod(0o755)

    import os
    monkeypatch.setenv("PATH", f"{untrusted_dir}:{os.environ.get('PATH', '')}")

    config = DynamicsConfig.model_construct(
        project_root=str(tmp_path),
        eon_binary="eonclient",
        trusted_directories=[str(dummy_bin_dir)],  # Only 'bin' is trusted
        eon_job="dummy",
        eon_min_mode_method="dummy",
    )
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = EONWrapper(config, sys_config)

    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="Resolved binary must reside in a trusted directory"):
        engine.run_kmc(None, work_dir)


def test_run_kmc_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dummy_bin_dir = tmp_path / "bin"
    dummy_bin_dir.mkdir(parents=True, exist_ok=True)
    dummy_eon = dummy_bin_dir / "eonclient"
    # Actually sleep to hit the timeout naturally
    dummy_eon.write_text("#!/bin/bash\ntrap '' TERM\nsleep 2\n")
    dummy_eon.chmod(0o755)

    import os
    monkeypatch.setenv("PATH", f"{dummy_bin_dir}:{os.environ.get('PATH', '')}")

    config = DynamicsConfig(
        project_root=str(tmp_path),
        eon_binary="eonclient",
        trusted_directories=[str(dummy_bin_dir)],
        eon_job="dummy",
        eon_min_mode_method="dummy",
    )
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = EONWrapper(config, sys_config)

    # We mock Popen just so we can force a quick timeout exception directly
    # out of communicate() to trigger the TimeoutExpired handling block.
    import subprocess

    class MockPopen:
        def __init__(self, *args, **kwargs):
            self.returncode = None
            self.killed = False
        def communicate(self, timeout=None):
            if not self.killed:
                # Force the timeout block to fire with the correct exception signature
                raise subprocess.TimeoutExpired(cmd=["eonclient"], timeout=timeout)
            return b"stdout", b"stderr"
        def kill(self):
            self.killed = True
        def wait(self):
            pass
        def poll(self):
            return None
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    monkeypatch.setattr(subprocess, "Popen", MockPopen)

    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match="EON client execution timed out"):
        engine.run_exploration(None, work_dir)
