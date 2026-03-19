import subprocess
from pathlib import Path
from typing import Any

import pytest

from src.domain_models.config import DynamicsConfig, SystemConfig
from src.dynamics.dynamics_engine import MDInterface


class SubprocessMockLAMMPS:
    """A realistic test double that simulates LAMMPS execution."""

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def __call__(
        self, cmd: list[str], *args: Any, **kwargs: Any
    ) -> subprocess.CompletedProcess[bytes]:
        work_dir = Path(kwargs.get("cwd", "."))

        # Validate LAMMPS command arguments
        if "-in" not in cmd:
            msg = "Missing required argument '-in' in LAMMPS command."
            raise ValueError(msg)

        in_idx = cmd.index("-in") + 1
        if in_idx >= len(cmd) or not cmd[in_idx].endswith(".lammps"):
            msg = "Invalid or missing input script argument in LAMMPS command."
            raise ValueError(msg)

        # Extract the input script and log file arguments using parsing
        log_idx = cmd.index("-log") + 1 if "-log" in cmd else -1
        log_file = work_dir / Path(cmd[log_idx]).name if log_idx > 0 else work_dir / "log.lammps"

        dump_file = work_dir / "dump.lammps"

        if self.mode == "halt":
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
            raise subprocess.CalledProcessError(1, cmd, output=b"", stderr=b"")

        if self.mode == "safe":
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
            log_file.write_text("NORMAL RUN\n")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")

        msg = "Unknown mode"
        raise ValueError(msg)


def test_uat_05_01_otf_halting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    UAT-05-01: On-The-Fly (OTF) Extrapolation Halting
    """
    import shutil

    mock_bin_dir = tmp_path / "bin"
    mock_bin_dir.mkdir(parents=True, exist_ok=True)
    mock_lmp = mock_bin_dir / "lmp"
    mock_lmp.touch()
    mock_lmp.chmod(0o755)

    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: str(mock_lmp.resolve()))

    from src.domain_models.config import ActiveLearningThresholds

    config = DynamicsConfig.model_construct(
        project_root=str(tmp_path),
        md_steps=1000,
        lmp_binary="lmp",
        trusted_directories=[str(mock_bin_dir)],
        thresholds=ActiveLearningThresholds(
            threshold_call_dft=5.0, threshold_add_train=0.02, smooth_steps=3
        ),
    )
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config)

    # Setup working directory
    work_dir = tmp_path / "md_run"

    # Define a proper test double
    mock_lammps = SubprocessMockLAMMPS(mode="halt")
    monkeypatch.setattr(subprocess, "run", mock_lammps)

    # Mock dummy potential
    pot_file = tmp_path / "dummy.yace"
    pot_file.touch()

    # Run exploration -> invokes actual SUT logic
    res = engine.run_exploration(potential=pot_file, work_dir=work_dir)

    dump_file = work_dir / "dump.lammps"

    # Verify the generated input script contains the watchdog instruction
    in_file = work_dir / "in.lammps"
    script_content = in_file.read_text()

    # Check the AL_HALT watchdog is correctly implemented without coupling to exact formatting
    assert "fix watchdog" in script_content
    assert "halt 3" in script_content
    assert "v_max_gamma > 5.0" in script_content
    assert "AL_HALT" in script_content

    # Verify parse result
    assert res["halted"] is True
    assert res["dump_file"] == dump_file


def test_uat_05_02_hybrid_potential_safety(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    UAT-05-02: Hybrid Potential Safety Enforcement
    """
    import shutil

    mock_bin_dir = tmp_path / "bin"
    mock_bin_dir.mkdir(parents=True, exist_ok=True)
    mock_lmp = mock_bin_dir / "lmp"
    mock_lmp.touch()
    mock_lmp.chmod(0o755)

    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: str(mock_lmp.resolve()))

    config = DynamicsConfig.model_construct(
        project_root=str(tmp_path), lmp_binary="lmp", trusted_directories=[str(mock_bin_dir)]
    )
    sys_config = SystemConfig(elements=["Fe", "Pt"], baseline_potential="zbl")
    engine = MDInterface(config, sys_config)

    work_dir = tmp_path / "md_run_2"
    work_dir.mkdir(parents=True)

    mock_lammps = SubprocessMockLAMMPS(mode="safe")
    monkeypatch.setattr(subprocess, "run", mock_lammps)

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
    from src.dynamics.security_utils import validate_executable_path

    # GIVEN maliciously set lmp_binary trying to traverse
    with pytest.raises(ValueError, match="Invalid characters in executable name"):
        validate_executable_path(
            executable_name="../usr/bin/lmp",
            trusted_directories=["/opt/mlip_bin"],
            project_root=str(tmp_path),
        )

    with pytest.raises(ValueError, match="Executable name cannot be absolute path"):
        validate_executable_path(
            executable_name="/usr/bin/lmp",
            trusted_directories=["/opt/mlip_bin"],
            project_root=str(tmp_path),
        )
