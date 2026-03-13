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


def test_eon_wrapper_ini_creation(tmp_path: Path, config: DynamicsConfig, sys_config: SystemConfig) -> None:
    wrapper = EONWrapper(config, sys_config)
    wrapper._write_config_ini(tmp_path)
    ini_file = tmp_path / "config.ini"
    assert ini_file.exists()
    content = ini_file.read_text()
    assert "[Process Search]" in content
    assert "min_mode_method = dimer" in content
    assert "job = process_search" in content


def test_eon_wrapper_driver_creation(tmp_path: Path, config: DynamicsConfig, sys_config: SystemConfig) -> None:
    wrapper = EONWrapper(config, sys_config)
    potential = tmp_path / "potential.yace"
    potential.touch()
    wrapper._write_pace_driver(tmp_path, potential)
    driver_file = tmp_path / "potentials" / "pace_driver.py"
    assert driver_file.exists()

    # Must be executable
    import os
    assert os.access(driver_file, os.X_OK)


def test_eon_wrapper_run_kmc_halt_mock(tmp_path: Path, config: DynamicsConfig, sys_config: SystemConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    # Use the mock hook built into EONWrapper for testing
    monkeypatch.setenv("MOCK_EON_HALT", "1")
    wrapper = EONWrapper(config, sys_config)

    potential = tmp_path / "potential.yace"
    potential.touch()

    res = wrapper.run_kmc(potential, tmp_path)

    assert res.get("halted") is True
    assert res.get("is_kmc") is True
    assert "dump_file" in res
    assert Path(res["dump_file"]).exists()


def test_eon_wrapper_run_kmc_no_halt(tmp_path: Path, config: DynamicsConfig, sys_config: SystemConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    # Use the mock hook to not halt
    monkeypatch.setenv("MOCK_EON_HALT", "0")

    # We mock shutil.which so it acts like eonclient is not found, letting the FileNotFoundError block handle it
    import shutil
    def mock_which(name: str) -> str | None:
        return None
    monkeypatch.setattr(shutil, "which", mock_which)

    wrapper = EONWrapper(config, sys_config)
    res = wrapper.run_kmc(None, tmp_path)
    assert res.get("halted") is False
