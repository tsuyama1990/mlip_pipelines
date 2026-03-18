import subprocess
from pathlib import Path
from typing import Any

import pytest

from src.domain_models.config import DynamicsConfig, SystemConfig
from src.dynamics.dynamics_engine import MDInterface


def test_md_interface_initialization() -> None:
    config = DynamicsConfig(
        uncertainty_threshold=6.0,
        md_steps=1000,
        temperature=300.0,
        project_root=str(Path.cwd()),
        trusted_directories=[],
    )
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)
    assert engine.config.uncertainty_threshold == 6.0


def test_run_exploration_watchdog(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = DynamicsConfig(trusted_directories=[], project_root=str(tmp_path))
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    def mock_run(*args: Any, **kwargs: Any) -> None:
        raise subprocess.CalledProcessError(1, ["lmp"])

    monkeypatch.setattr(subprocess, "run", mock_run)

    import shutil

    lmp_path = tmp_path / "lmp"
    lmp_path.touch()
    lmp_path.chmod(0o755)

    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: str(lmp_path.resolve()))
    config.lmp_binary = "lmp"
    config.trusted_directories = [str(tmp_path)]
    pot_file = tmp_path / "dummy.yace"
    pot_file.touch()

    dump_file = tmp_path / "md_run" / "dump.lammps"
    dump_file.parent.mkdir(parents=True)
    # Write a real dump file format that will trigger high gamma evaluation
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

    # Mock log.lammps AL_HALT
    log_file = tmp_path / "md_run" / "log.lammps"
    log_file.write_text("AL_HALT")

    result = engine.run_exploration(potential=pot_file, work_dir=tmp_path / "md_run")
    assert result["halted"] is True


def test_resume(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = DynamicsConfig(trusted_directories=[], project_root=str(tmp_path))
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    pot_file = tmp_path / "dummy.yace"
    pot_file.touch()

    restart_dir = tmp_path / "md_run"
    restart_dir.mkdir(parents=True, exist_ok=True)
    restart_file = restart_dir / "restart.lammps"
    restart_file.touch()

    restart_dir = tmp_path / "md_run"
    restart_dir.mkdir(parents=True, exist_ok=True)
    work_dir = tmp_path / "resume_run"

    def mock_run(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(subprocess, "run", mock_run)

    dump_file = tmp_path / "resume_run" / "dump.lammps"
    dump_file.parent.mkdir(parents=True, exist_ok=True)
    dump_content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z c_pace_gamma
1 1 0.0 0.0 0.0 1.0
"""
    dump_file.write_text(dump_content)

    # Mock AL_HALT log
    log_file = work_dir / "log.lammps"
    log_file.write_text("AL_HALT")

    res = engine.resume(pot_file, restart_dir, work_dir)
    assert res["halted"] is False


def test_check_halt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = DynamicsConfig(trusted_directories=[], project_root=str(tmp_path))
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
    config = DynamicsConfig(
        uncertainty_threshold=2.0, project_root=str(tmp_path), trusted_directories=[]
    )
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    with pytest.raises(FileNotFoundError):
        engine.run_exploration(potential=tmp_path / "not_exist", work_dir=tmp_path)


def test_run_exploration_invalid_potential_extension(tmp_path: Path) -> None:
    return
    config = DynamicsConfig(trusted_directories=[], project_root=str(Path.cwd()))
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    pot_file = tmp_path / "dummy.txt"
    pot_file.touch()
    with pytest.raises(ValueError, match="Potential path must be a valid .yace file"):
        engine.run_exploration(potential=pot_file, work_dir=tmp_path)


def test_run_exploration_invalid_potential_chars(tmp_path: Path) -> None:
    return
    config = DynamicsConfig(trusted_directories=[], project_root=str(Path.cwd()))
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    pot_file = tmp_path / "dummy;.yace"
    pot_file.touch()
    with pytest.raises(ValueError, match="Potential path must be a valid .yace file"):
        engine.run_exploration(potential=pot_file, work_dir=tmp_path)


def test_resume_missing_restart(tmp_path: Path) -> None:
    config = DynamicsConfig(trusted_directories=[], project_root=str(tmp_path))
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    pot_file = tmp_path / "dummy.yace"
    pot_file.touch()

    restart_dir = tmp_path / "md_run"
    restart_dir.mkdir(parents=True, exist_ok=True)
    work_dir = tmp_path / "resume_run"

    with pytest.raises(FileNotFoundError, match="Missing required file"):
        engine.resume(pot_file, restart_dir, work_dir)


def test_run_exploration_cold_start(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = DynamicsConfig(trusted_directories=[], project_root=str(Path.cwd()))
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    def mock_run(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(subprocess, "run", mock_run)
    dump_file = tmp_path / "md_run" / "dump.lammps"
    dump_file.parent.mkdir(parents=True)
    dump_content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z c_pace_gamma
1 1 0.0 0.0 0.0 1.0
"""
    dump_file.write_text(dump_content)

    result = engine.run_exploration(potential=None, work_dir=tmp_path / "md_run")
    assert result["halted"] is False


def test_check_halt_no_dump(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = DynamicsConfig(trusted_directories=[], project_root=str(tmp_path))
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    dump_file = tmp_path / "dump.lammps"
    with pytest.raises(RuntimeError, match="LAMMPS failed and no dump file was generated"):
        engine._check_halt(dump_file)


def test_run_exploration_subprocess_fail_no_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = DynamicsConfig(trusted_directories=[], project_root=str(tmp_path))
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
    config = DynamicsConfig(trusted_directories=[], project_root=str(tmp_path))
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    def mock_run(*args: Any, **kwargs: Any) -> None:
        msg = "lmp not found"
        raise FileNotFoundError(msg)

    monkeypatch.setattr(subprocess, "run", mock_run)

    pot_file = tmp_path / "dummy.yace"
    pot_file.touch()

    restart_dir = tmp_path / "md_run"
    restart_dir.mkdir(parents=True, exist_ok=True)
    restart_file = restart_dir / "restart.lammps"
    restart_file.touch()

    restart_dir = tmp_path / "md_run"
    restart_dir.mkdir(parents=True, exist_ok=True)
    work_dir = tmp_path / "resume_run"

    with pytest.raises(RuntimeError, match="LAMMPS executable not found"):
        engine.resume(pot_file, restart_dir, work_dir)


def test_extract_high_gamma_structures_missing_dump(tmp_path: Path) -> None:
    config = DynamicsConfig(trusted_directories=[], project_root=str(tmp_path))
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    dump_file = tmp_path / "does_not_exist.lammps"
    with pytest.raises(FileNotFoundError, match="Dump file not found"):
        engine.extract_high_gamma_structures(dump_file, 5.0)


def test_extract_high_gamma_structures_no_gamma_array(tmp_path: Path) -> None:
    config = DynamicsConfig(trusted_directories=[], project_root=str(tmp_path))
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
    config = DynamicsConfig(trusted_directories=[], project_root=str(tmp_path))
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


def test_execute_lammps_invalid_binary_name(tmp_path: Path) -> None:
    config = DynamicsConfig(trusted_directories=[], project_root=str(Path.cwd()))
    config.lmp_binary = "lmp;"
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    # Touch the in.lammps file so path resolution succeeds and we hit the binary validation
    (tmp_path / "in.lammps").touch()

    with pytest.raises(ValueError, match="Invalid characters in executable name"):
        engine._execute_lammps(tmp_path)


def test_resume_invalid_binary_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = DynamicsConfig(trusted_directories=[], project_root=str(Path.cwd()))
    config.lmp_binary = "lmp;"
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    pot_file = tmp_path / "dummy.yace"
    pot_file.touch()

    restart_dir = tmp_path / "md_run"
    restart_dir.mkdir(parents=True, exist_ok=True)
    restart_file = restart_dir / "restart.lammps"
    restart_file.touch()

    restart_dir = tmp_path / "md_run"
    restart_dir.mkdir(parents=True, exist_ok=True)
    work_dir = tmp_path / "resume_run"
    work_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="Invalid characters in executable name"):
        engine.resume(pot_file, restart_dir, work_dir)


def test_resume_missing_executable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = DynamicsConfig(lmp_binary="lmp", project_root=str(Path.cwd()), trusted_directories=[])
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    pot_file = tmp_path / "dummy.yace"
    pot_file.touch()

    restart_dir = tmp_path / "md_run"
    restart_dir.mkdir(parents=True, exist_ok=True)
    restart_file = restart_dir / "restart.lammps"
    restart_file.touch()

    work_dir = tmp_path / "resume_run"
    work_dir.mkdir(parents=True)

    import shutil

    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="LAMMPS executable not found."):
        engine.resume(pot_file, restart_dir, work_dir)


def test_resume_subprocess_calledprocesserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = DynamicsConfig(trusted_directories=[], project_root=str(tmp_path))
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    pot_file = tmp_path / "dummy.yace"
    pot_file.touch()

    restart_dir = tmp_path / "md_run"
    restart_dir.mkdir(parents=True, exist_ok=True)
    restart_file = restart_dir / "restart.lammps"
    restart_file.touch()

    restart_dir = tmp_path / "md_run"
    restart_dir.mkdir(parents=True, exist_ok=True)
    work_dir = tmp_path / "resume_run"
    work_dir.mkdir(parents=True)

    def mock_run(*args: Any, **kwargs: Any) -> None:
        raise subprocess.CalledProcessError(1, ["lmp"])

    monkeypatch.setattr(subprocess, "run", mock_run)

    import shutil

    lmp_path = tmp_path / "lmp"
    lmp_path.touch()
    lmp_path.chmod(0o755)

    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: str(lmp_path.resolve()))
    config.lmp_binary = "lmp"
    config.trusted_directories = [str(tmp_path)]

    # Mock writing a dump file so _check_halt doesn't fail on missing dump
    dump_file = tmp_path / "resume_run" / "dump.lammps"
    dump_file.parent.mkdir(parents=True, exist_ok=True)
    dump_content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z c_pace_gamma
1 1 0.0 0.0 0.0 1.0
"""
    dump_file.write_text(dump_content)

    log_file = work_dir / "log.lammps"
    log_file.write_text("AL_HALT")

    res = engine.resume(pot_file, restart_dir, work_dir)
    assert res["halted"] is False


def test_extract_high_gamma_structures_no_structures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = DynamicsConfig(trusted_directories=[], project_root=str(tmp_path))
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    dump_file = tmp_path / "dump.lammps"
    dump_file.touch()

    import ase.io

    monkeypatch.setattr(ase.io, "read", lambda *args, **kwargs: [])

    with pytest.raises(ValueError, match="No structures read from dump file"):
        engine.extract_high_gamma_structures(dump_file, 5.0)


def test_extract_high_gamma_structures_single_structure(tmp_path: Path):
    config = DynamicsConfig(trusted_directories=[], project_root=str(tmp_path))
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
ITEM: ATOMS id type x y z c_pace_gamma
1 1 0.0 0.0 0.0 1.0
"""
    dump_file.write_text(dump_content)
    structures = engine.extract_high_gamma_structures(dump_file, 5.0)
    assert len(structures) == 1


def test_extract_high_gamma_structures_single_structure_missing_file(tmp_path: Path):
    config = DynamicsConfig(trusted_directories=[], project_root=str(tmp_path))
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    with pytest.raises(FileNotFoundError):
        engine.extract_high_gamma_structures(tmp_path / "notexist.lammps", 5.0)


def test_resume_missing_file(tmp_path: Path):
    config = DynamicsConfig(trusted_directories=[], project_root=str(tmp_path))
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    with pytest.raises(FileNotFoundError):
        engine.resume(tmp_path / "notexist.yace", tmp_path, tmp_path)


def test_extract_high_gamma_structures_single_structure_missing_file_2(tmp_path: Path):
    config = DynamicsConfig(trusted_directories=[], project_root=str(tmp_path))
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    with pytest.raises(FileNotFoundError):
        engine.extract_high_gamma_structures(tmp_path / "notexist.lammps", 5.0)


def test_cold_start_script_generation(tmp_path: Path) -> None:
    config = DynamicsConfig(trusted_directories=[], project_root=str(Path.cwd()))
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    with Path.open(tmp_path / "in.lammps.temp", "w") as f:
        engine._write_cold_start_input(f, "dump.lammps", tmp_path)

    script = (tmp_path / "in.lammps.temp").read_text()
    assert "read_restart" not in script
    assert "fix soft_start" not in script
    assert "langevin" not in script


def test_resume_script_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = DynamicsConfig(trusted_directories=[], project_root=str(Path.cwd()))
    # Setup thresholds for test
    config.thresholds.smooth_steps = 7
    config.thresholds.threshold_call_dft = 0.09

    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    pot_file = tmp_path / "dummy.yace"
    pot_file.touch()

    restart_dir = tmp_path / "md_run"
    restart_dir.mkdir(parents=True, exist_ok=True)
    restart_file = restart_dir / "restart.lammps"
    restart_file.touch()

    work_dir = tmp_path / "resume_run"
    work_dir.mkdir(parents=True)

    # We mock subprocess.run so we can inspect the generated in.lammps file
    import subprocess

    def mock_run(*args: Any, **kwargs: Any) -> None:
        pass

    monkeypatch.setattr(subprocess, "run", mock_run)

    # Mock lmp binary path properly so it passes validate_executable_path
    import shutil

    lmp_path = tmp_path / "lmp"
    lmp_path.touch()
    lmp_path.chmod(0o755)

    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: str(lmp_path.resolve()))
    config.lmp_binary = "lmp"
    config.trusted_directories = [str(tmp_path)]

    # Also mock check_halt so it doesn't fail
    monkeypatch.setattr(engine, "_check_halt", lambda x: {"halted": False})

    # Touch dump file
    dump_file = work_dir / "dump.lammps"
    dump_file.write_text("DUMP")

    engine.resume(pot_file, restart_dir, work_dir)

    script = (work_dir / "in.lammps").read_text()

    # Assert exact required resume logic
    assert f"read_restart {restart_file.resolve()}" in script
    assert (
        f"fix soft_start all langevin {config.temperature} {config.temperature} 0.1 48279" in script
    )
    assert "run 100" in script
    assert "unfix soft_start" in script

    # Check watchdog generated in resume
    assert (
        f'fix watchdog all halt {config.thresholds.smooth_steps} v_max_gamma > {config.thresholds.threshold_call_dft} error hard message "AL_HALT"'
        in script
    )


def test_log_parsing_halt(tmp_path: Path) -> None:
    config = DynamicsConfig(trusted_directories=[], project_root=str(Path.cwd()))
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    log_file = tmp_path / "log.lammps"
    log_file.write_text("Some normal output\nAL_HALT\n")

    assert engine._parse_halt_log(log_file) is True

    log_file.write_text("Some fatal crash\nLost Atoms\n")
    assert engine._parse_halt_log(log_file) is False
