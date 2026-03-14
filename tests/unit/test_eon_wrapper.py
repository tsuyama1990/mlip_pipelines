from pathlib import Path

import pytest

from src.domain_models.config import DynamicsConfig, SystemConfig
from src.dynamics.eon_wrapper import EONWrapper


@pytest.fixture
def config() -> DynamicsConfig:
    return DynamicsConfig(
        uncertainty_threshold=5.0,
        md_steps=100,
        temperature=300.0,
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
    config = DynamicsConfig()
    config.eon_job = "job;"
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = EONWrapper(config, sys_config)
    with pytest.raises(ValueError, match="Invalid EON job string"):
        engine._write_config_ini(tmp_path)

def test_write_config_ini_invalid_min_mode_method(tmp_path: Path) -> None:
    config = DynamicsConfig()
    config.eon_min_mode_method = "method;"
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = EONWrapper(config, sys_config)
    with pytest.raises(ValueError, match="Invalid EON min_mode_method string"):
        engine._write_config_ini(tmp_path)

def test_write_pace_driver_invalid_potential(tmp_path: Path) -> None:
    config = type("MockConfig", (), {"project_root": str(tmp_path), "eon_binary": "eonclient", "trusted_directories": [], "eon_job": "dummy", "eon_min_mode_method": "dummy", "temperature": 300, "eon_config_template": "{eon_job} {temperature} {eon_min_mode_method}", "eon_driver_template": "{executable} {threshold} {pot_str}", "uncertainty_threshold": 5.0})()
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = EONWrapper(config, sys_config)

    # Potential path is outside project root
    pot_path = Path("/var/tmp/hacker_dummy.yace")  # noqa: S108
    pot_path.parent.mkdir(parents=True, exist_ok=True)
    pot_path.touch()

    with pytest.raises(ValueError, match="Potential path must be within the project root"):
        engine._write_pace_driver(tmp_path, pot_path)

def test_write_pace_driver_invalid_python_executable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = DynamicsConfig()
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = EONWrapper(config, sys_config)

    import sys
    monkeypatch.setattr(sys, "executable", "python;")

    with pytest.raises(ValueError, match="Invalid python executable path"):
        engine._write_pace_driver(tmp_path, None)

def test_run_kmc_invalid_work_dir(tmp_path: Path) -> None:
    config = type("MockConfig", (), {"project_root": str(tmp_path), "eon_binary": "eonclient", "trusted_directories": [], "eon_job": "dummy", "eon_min_mode_method": "dummy", "temperature": 300, "eon_config_template": "{eon_job} {temperature} {eon_min_mode_method}", "eon_driver_template": "{executable} {threshold} {pot_str}", "uncertainty_threshold": 5.0})()
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = EONWrapper(config, sys_config)

    work_dir = Path("/var/tmp/hacker_work")  # noqa: S108
    with pytest.raises(ValueError, match="is outside the allowed project root"):
        engine.run_kmc(None, work_dir)

def test_run_kmc_subprocess_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = type("MockConfig", (), {"project_root": str(tmp_path), "eon_binary": "eonclient", "trusted_directories": [], "eon_job": "dummy", "eon_min_mode_method": "dummy", "temperature": 300, "eon_config_template": "{eon_job} {temperature} {eon_min_mode_method}", "eon_driver_template": "{executable} {threshold} {pot_str}", "uncertainty_threshold": 5.0})()
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = EONWrapper(config, sys_config)

    dummy = tmp_path / "bin" / "eonclient"
    dummy.parent.mkdir(parents=True, exist_ok=True)
    dummy.touch()
    dummy.chmod(0o755)

    import shutil
    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: str(dummy))

    class MockRes:
        returncode = 1

    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: MockRes())

    with pytest.raises(RuntimeError, match="EON client failed with return code"):
        engine.run_kmc(None, tmp_path / "work")

def test_run_kmc_subprocess_halted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = type("MockConfig", (), {"project_root": str(tmp_path), "eon_binary": "eonclient", "trusted_directories": [], "eon_job": "dummy", "eon_min_mode_method": "dummy", "temperature": 300, "eon_config_template": "{eon_job} {temperature} {eon_min_mode_method}", "eon_driver_template": "{executable} {threshold} {pot_str}", "uncertainty_threshold": 5.0})()
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = EONWrapper(config, sys_config)

    dummy = tmp_path / "bin" / "eonclient"
    dummy.parent.mkdir(parents=True, exist_ok=True)
    dummy.touch()
    dummy.chmod(0o755)

    import shutil
    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: str(dummy))

    class MockRes:
        returncode = 100

    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: MockRes())

    res = engine.run_kmc(None, tmp_path / "work")
    assert res["halted"] is True
    assert res["is_kmc"] is True

def test_run_kmc_missing_executable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = type("MockConfig", (), {"project_root": str(tmp_path), "eon_binary": "eonclient", "trusted_directories": [], "eon_job": "dummy", "eon_min_mode_method": "dummy", "temperature": 300, "eon_config_template": "{eon_job} {temperature} {eon_min_mode_method}", "eon_driver_template": "{executable} {threshold} {pot_str}", "uncertainty_threshold": 5.0})()
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = EONWrapper(config, sys_config)

    import src.dynamics.security_utils
    def mock_val(*args, **kwargs):
        msg = "EON client executable not found."
        raise RuntimeError(msg)

    monkeypatch.setattr(src.dynamics.security_utils, "validate_executable_path", mock_val)

    with pytest.raises(RuntimeError, match="EON client executable not found."):
        engine.run_kmc(None, tmp_path / "work")
