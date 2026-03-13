import subprocess
from pathlib import Path
from typing import Any

import pytest

from src.domain_models.config import DynamicsConfig, SystemConfig
from src.dynamics.dynamics_engine import MDInterface


def test_md_interface_initialization() -> None:
    config = DynamicsConfig(uncertainty_threshold=6.0, md_steps=1000, temperature=300.0)
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)
    assert engine.config.uncertainty_threshold == 6.0


def test_run_exploration_watchdog(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = DynamicsConfig()
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    def mock_run(*args: Any, **kwargs: Any) -> None:
        raise subprocess.CalledProcessError(1, ["lmp"])

    monkeypatch.setattr(subprocess, "run", mock_run)
    pot_file = tmp_path / "dummy.yace"
    pot_file.touch()

    dump_file = tmp_path / "md_run" / "dump.lammps"
    dump_file.parent.mkdir(parents=True)
    dump_file.touch()

    def mock_check_halt(self: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"halted": True, "dump_file": dump_file}

    monkeypatch.setattr(MDInterface, "_check_halt", mock_check_halt)

    result = engine.run_exploration(potential=pot_file, work_dir=tmp_path / "md_run")
    assert result["halted"] is True


def test_resume(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = DynamicsConfig()
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    pot_file = tmp_path / "dummy.yace"
    pot_file.touch()

    restart_dir = tmp_path / "md_run"
    restart_dir.mkdir(parents=True)
    restart_file = restart_dir / "restart.lammps"
    restart_file.touch()

    work_dir = tmp_path / "resume_run"

    def mock_run(*args: Any, **kwargs: Any) -> None:
        pass

    monkeypatch.setattr(subprocess, "run", mock_run)

    def mock_check_halt(self: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"halted": False, "dump_file": None}

    monkeypatch.setattr(MDInterface, "_check_halt", mock_check_halt)

    res = engine.resume(pot_file, restart_dir, work_dir)
    assert res["halted"] is False


def test_check_halt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = DynamicsConfig()
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    dump_file = tmp_path / "dump.lammps"
    # Basic LAMMPS dump content with 1 atom
    dump_content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z c_pace_gamma
1 1 0.0 0.0 0.0 6.0
"""
    dump_file.write_text(dump_content)

    res = engine._check_halt(dump_file)
    assert res["halted"] is True


def test_run_exploration_invalid_potential(tmp_path: Path) -> None:
    config = DynamicsConfig()
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    with pytest.raises(FileNotFoundError):
        engine.run_exploration(potential=tmp_path / "not_exist", work_dir=tmp_path)


def test_run_exploration_invalid_potential_extension(tmp_path: Path) -> None:
    config = DynamicsConfig()
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    pot_file = tmp_path / "dummy.txt"
    pot_file.touch()
    with pytest.raises(ValueError, match="Potential path must end with .yace"):
        engine.run_exploration(potential=pot_file, work_dir=tmp_path)


def test_run_exploration_invalid_potential_chars(tmp_path: Path) -> None:
    config = DynamicsConfig()
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    pot_file = tmp_path / "dummy;.yace"
    pot_file.touch()
    with pytest.raises(ValueError, match="Potential path contains invalid characters"):
        engine.run_exploration(potential=pot_file, work_dir=tmp_path)


def test_resume_missing_restart(tmp_path: Path) -> None:
    config = DynamicsConfig()
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    pot_file = tmp_path / "dummy.yace"
    pot_file.touch()

    restart_dir = tmp_path / "md_run"
    restart_dir.mkdir(parents=True)
    work_dir = tmp_path / "resume_run"

    with pytest.raises(FileNotFoundError, match="Missing required file"):
        engine.resume(pot_file, restart_dir, work_dir)

def test_run_exploration_cold_start(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = DynamicsConfig()
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    def mock_run(*args: Any, **kwargs: Any) -> None:
        pass

    monkeypatch.setattr(subprocess, "run", mock_run)
    dump_file = tmp_path / "md_run" / "dump.lammps"
    dump_file.parent.mkdir(parents=True)
    dump_file.touch()

    def mock_check_halt(self: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"halted": False, "dump_file": dump_file}

    monkeypatch.setattr(MDInterface, "_check_halt", mock_check_halt)

    result = engine.run_exploration(potential=None, work_dir=tmp_path / "md_run")
    assert result["halted"] is False


def test_check_halt_no_dump(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = DynamicsConfig()
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    dump_file = tmp_path / "dump.lammps"
    with pytest.raises(RuntimeError, match="LAMMPS failed and no dump file was generated"):
        engine._check_halt(dump_file)


def test_run_exploration_subprocess_fail_no_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = DynamicsConfig()
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    def mock_run(*args: Any, **kwargs: Any) -> None:
        msg = "lmp not found"
        raise FileNotFoundError(msg)

    monkeypatch.setattr(subprocess, "run", mock_run)

    pot_file = tmp_path / "dummy.yace"
    pot_file.touch()

    with pytest.raises(RuntimeError, match="LAMMPS executable not found"):
        engine.run_exploration(potential=pot_file, work_dir=tmp_path / "md_run")


def test_resume_subprocess_fail_no_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = DynamicsConfig()
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    def mock_run(*args: Any, **kwargs: Any) -> None:
        msg = "lmp not found"
        raise FileNotFoundError(msg)

    monkeypatch.setattr(subprocess, "run", mock_run)

    pot_file = tmp_path / "dummy.yace"
    pot_file.touch()

    restart_dir = tmp_path / "md_run"
    restart_dir.mkdir(parents=True)
    restart_file = restart_dir / "restart.lammps"
    restart_file.touch()

    work_dir = tmp_path / "resume_run"

    with pytest.raises(RuntimeError, match="LAMMPS executable not found"):
        engine.resume(pot_file, restart_dir, work_dir)

def test_extract_high_gamma_structures_missing_dump(tmp_path: Path) -> None:
    config = DynamicsConfig()
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    dump_file = tmp_path / "does_not_exist.lammps"
    with pytest.raises(FileNotFoundError, match="Dump file not found"):
        engine.extract_high_gamma_structures(dump_file, 5.0)

def test_extract_high_gamma_structures_no_gamma_array(tmp_path: Path) -> None:
    config = DynamicsConfig()
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    dump_file = tmp_path / "dump.lammps"
    dump_content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z
1 1 0.0 0.0 0.0
"""
    dump_file.write_text(dump_content)
    structures = engine.extract_high_gamma_structures(dump_file, 5.0)
    assert len(structures) == 1

def test_extract_high_gamma_structures(tmp_path: Path) -> None:
    config = DynamicsConfig()
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    dump_file = tmp_path / "dump.lammps"
    # Basic LAMMPS dump content with 1 atom
    dump_content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z
1 1 0.0 0.0 0.0
"""
    dump_file.write_text(dump_content)

    structures = engine.extract_high_gamma_structures(dump_file, 5.0)
    assert isinstance(structures, list)
    assert len(structures) == 1
    assert len(structures[0]) == 1
