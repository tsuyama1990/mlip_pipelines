import subprocess
from pathlib import Path
from typing import Any

import pytest

from src.domain_models.config import DynamicsConfig, SystemConfig
from src.dynamics.dynamics_engine import MDInterface


def test_uat_05_01_otf_halting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    UAT-05-01: On-The-Fly (OTF) Extrapolation Halting
    """
    from src.domain_models.config import ActiveLearningThresholds
    config = DynamicsConfig.model_construct(
        project_root=str(tmp_path),
        md_steps=1000,
        lmp_binary="lmp",
        trusted_directories=[str(tmp_path)],
        thresholds=ActiveLearningThresholds(threshold_call_dft=5.0, threshold_add_train=0.02, smooth_steps=3)
    )
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    # Mock security_utils.validate_executable_path to prevent executable file checks
    from unittest.mock import patch
    monkeypatch.setattr("src.dynamics.security_utils.validate_executable_path", lambda *args, **kwargs: tmp_path / "lmp")

    # Setup working directory
    work_dir = tmp_path / "md_run"
    work_dir.mkdir(parents=True)
    dump_file = work_dir / "dump.lammps"
    log_file = work_dir / "log.lammps"

    # Define a test double for subprocess.run
    def mock_run(cmd: list[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess:
        # Simulate LAMMPS output dynamically
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
        log_file.write_text("AL_HALT\n")
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "run", mock_run)

    # Mock dummy potential
    pot_file = tmp_path / "dummy.yace"
    pot_file.touch()

    # Run exploration
    res = engine.run_exploration(potential=pot_file, work_dir=work_dir)

    # Verify the generated input script contains the watchdog instruction
    in_file = work_dir / "in.lammps"
    script_content = in_file.read_text()

    # Check the AL_HALT watchdog is correctly implemented
    assert (
        'fix watchdog all halt 3 v_max_gamma > 5.0 error hard message "AL_HALT"' in script_content
    )

    # Verify parse result
    assert res["halted"] is True
    assert res["dump_file"] == dump_file


def test_uat_05_02_hybrid_potential_safety(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    UAT-05-02: Hybrid Potential Safety Enforcement
    """
    config = DynamicsConfig.model_construct(
        project_root=str(tmp_path),
        lmp_binary="lmp",
        trusted_directories=[str(tmp_path)]
    )
    sys_config = SystemConfig(elements=["Fe", "Pt"], baseline_potential="zbl")
    engine = MDInterface(config, sys_config)

    # Mock security_utils.validate_executable_path to prevent executable file checks
    monkeypatch.setattr("src.dynamics.security_utils.validate_executable_path", lambda *args, **kwargs: tmp_path / "lmp")

    work_dir = tmp_path / "md_run_2"
    work_dir.mkdir(parents=True)
    dump_file = work_dir / "dump.lammps"

    # Define a test double for subprocess.run
    def mock_run(cmd: list[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess:
        # Simulate LAMMPS output dynamically
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
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", mock_run)

    pot_file = tmp_path / "dummy.yace"
    pot_file.touch()

    engine.run_exploration(potential=pot_file, work_dir=work_dir)

    in_file = work_dir / "in.lammps"
    script_content = in_file.read_text()

    # Verify the script has hybrid overlay and pair coeff definitions
    assert "pair_style hybrid/overlay pace zbl 1.0 2.0" in script_content
    assert "pair_coeff * * pace" in script_content
    assert "pair_coeff * * zbl" in script_content
    # ZBL mapping for Fe (26) and Pt (78)
    assert "26 78" in script_content


def test_uat_05_03_secure_sandbox_execution(tmp_path: Path) -> None:
    """
    UAT-05-03: Secure Sandbox Path Execution
    """
    # GIVEN maliciously set lmp_binary
    with pytest.raises(
        ValueError, match="Binary name cannot contain path separators or traversal characters"
    ):
        DynamicsConfig(
            lmp_binary="../usr/bin/python3",
            trusted_directories=["/opt/mlip_bin"],
            project_root=str(tmp_path),
        )
