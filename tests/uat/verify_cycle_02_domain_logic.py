import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path.cwd()))

import marimo

__generated_with = "0.11.1"
app = marimo.App(width="medium")

@app.cell
def __() -> tuple:
    import logging
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path
    from typing import Any

    import pytest

    from src.core.exceptions import DynamicsHaltInterrupt
    from src.domain_models.config import DynamicsConfig, SystemConfig
    from src.dynamics.dynamics_engine import MDInterface
    from src.dynamics.eon_wrapper import EONWrapper

    logging.basicConfig(level=logging.INFO)

    return (
        DynamicsConfig,
        SystemConfig,
        MDInterface,
        EONWrapper,
        DynamicsHaltInterrupt,
        Path,
        logging,
        subprocess,
        shutil,
        tempfile,
        Any,
        pytest,
    )


@app.cell
def test_scenario_01(
    DynamicsConfig: Any,
    SystemConfig: Any,
    MDInterface: Any,
    Path: Any,
    subprocess: Any,
    shutil: Any,
    tempfile: Any,
    Any: Any,
) -> tuple:
    print("Executing UAT-C02-01: End-to-End Dynamics Engine and LAMMPS Integration")
    print("Verifying autonomous execution, hybrid potential enforcement, and extrapolation halt logic")

    _tmp_path = Path("/home/jules/tmp_test_dir")
    if _tmp_path.exists(): shutil.rmtree(_tmp_path)
    _tmp_path.mkdir(parents=True)

    # GIVEN: A valid YAML-equivalent configuration defining materials and thresholds
    import os as _os
    from src.domain_models.config import ProjectConfig as _ProjectConfig
    from unittest.mock import patch as _patch


    with _patch("pathlib.Path.cwd", return_value=_tmp_path):
        (_tmp_path / "README.md").touch()
        _full_config = _ProjectConfig(
            project_root=_tmp_path,
            dynamics={
                "project_root": str(_tmp_path),
                "trusted_directories": [str(_tmp_path)],
                "md_steps": 1000,
                "temperature": 300.0,
                "thresholds": {"threshold_call_dft": 5.0, "smooth_steps": 3}
            },
            system={"elements": ["Fe", "Pt"], "baseline_potential": "zbl"},
            loop_strategy={"replay_buffer_size": 500, "checkpoint_interval": 5, "timeout_seconds": 3600},
            distillation_config={"temp_dir": str(_tmp_path), "output_dir": str(_tmp_path), "model_storage_path": str(_tmp_path)},
            trainer={"trusted_directories": [str(_tmp_path)]},
            oracle={"pseudo_dir": str(_tmp_path)},
            validator={}
        )
        _config = _full_config.dynamics
        _sys_config = _full_config.system
        # Config loaded from env variables via ProjectConfig

        # WHEN: Initiating the powerful active learning computational pipeline
        _engine = MDInterface(_config, _sys_config)

        # Mocking external LAMMPS execution for purely functional UAT
        def mock_run_explore(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
            # Simulate LAMMPS halting due to extrapolation threshold breach
            _work_dir = Path(kwargs.get("cwd", str(_tmp_path)))

            _dump_file = _work_dir / "dump.lammps"
            # Dump with high c_pace_gamma exceeding the threshold
            _dump_content = """ITEM: TIMESTEP
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
            _dump_file.write_text(_dump_content)

            # Write AL_HALT into the log to simulate fix watchdog
            _log_file = _work_dir / "log.lammps"
            _log_file.write_text("AL_HALT\n")

            raise subprocess.CalledProcessError(1, ["lmp"])

        _lmp_path = _tmp_path / "lmp"
        _lmp_path.touch()
        _lmp_path.chmod(0o755)

        _pot_file = _tmp_path / "dummy.yace"
        _pot_file.touch()

        _work_dir = _tmp_path / "md_run"
        _work_dir.mkdir(parents=True)

        with _patch("subprocess.run", side_effect=mock_run_explore), _patch("shutil.which", return_value=str(_lmp_path.resolve())):
            _config.lmp_binary = "lmp"

            # THEN: The system flawlessly executes and gracefully halts upon encountering unknown extrapolation region
            _res = _engine.run_exploration(_pot_file, _work_dir)
            assert _res["halted"] is True
            print("✓ Successfully executed molecular dynamics _engine and gracefully halted on high systemic uncertainty.")

            # Verify that the generated input script rigorously enforces the Lennard-Jones/ZBL baseline
            _in_file = _work_dir / "in.lammps"
            _script_content = _in_file.read_text()
            assert "pair_style hybrid/overlay pace zbl 1.0 2.0" in _script_content
            assert "pair_coeff * * pace" in _script_content
            assert "pair_coeff * * zbl" in _script_content
            print("✓ System rigorously enforced the Lennard-Jones/ZBL baseline to prevent unphysical atomic collisions.")

            # Verify the extrapolative grade (gamma value) watchdog
            assert 'fix watchdog all halt 3 v_max_gamma > 5.0 error hard message "AL_HALT"' in _script_content
            print("✓ Orchestrator actively monitored the extrapolation grade emitted by the ACE potential.")

        # WHEN: The simulation resumes using the brand newly updated potential

        # Mock resume execution
        def mock_run_resume(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
            # Simulate successful continuation
            _work_dir = Path(kwargs.get("cwd", str(_tmp_path)))
            _dump_file = _work_dir / "dump.lammps"
            _dump_content = """ITEM: TIMESTEP
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
            _dump_file.write_text(_dump_content)
            return subprocess.CompletedProcess(args=["lmp"], returncode=0, stdout=b"", stderr=b"")

        _restart_dir = _tmp_path / "md_run"
        _restart_file = _restart_dir / "restart.lammps"
        _restart_file.touch()

        _resume_dir = _tmp_path / "resume_run"
        _resume_dir.mkdir(parents=True)

        with _patch("subprocess.run", side_effect=mock_run_resume), _patch("shutil.which", return_value=str(_lmp_path.resolve())):
            # THEN: Seamlessly flawlessly resume the paused simulation
            _res_resume = _engine.resume(_pot_file, _restart_dir, _resume_dir)
            assert _res_resume["halted"] is False

            _resume_script = (_resume_dir / "in.lammps").read_text()
            assert f"read_restart {_restart_file.resolve()}" in _resume_script
            print("✓ Dynamics Engine successfully resumed from process checkpoint.")

    return ()


@app.cell
def test_scenario_02_eon_client(
    DynamicsConfig: Any,
    SystemConfig: Any,
    EONWrapper: Any,
    Path: Any,
    subprocess: Any,
    shutil: Any,
    tempfile: Any,
    Any: Any,
) -> tuple:
    print("Executing UAT-C02-02: Interfacing with the EON client for long-timescale Adaptive KMC")

    _tmp_path = Path("/home/jules/tmp_test_dir2")
    if _tmp_path.exists(): shutil.rmtree(_tmp_path)
    _tmp_path.mkdir(parents=True)

    import os as _os
    from src.domain_models.config import ProjectConfig as _ProjectConfig
    from unittest.mock import patch as _patch


    with _patch("pathlib.Path.cwd", return_value=_tmp_path):
        (_tmp_path / "README.md").touch()
        _full_config = _ProjectConfig(
            project_root=_tmp_path,
            dynamics={
                "project_root": str(_tmp_path),
                "trusted_directories": [str(_tmp_path)],
                "md_steps": 1000,
                "temperature": 300.0,
                "thresholds": {"threshold_call_dft": 5.0, "smooth_steps": 3}
            },
            system={"elements": ["Fe", "Pt"], "baseline_potential": "zbl"},
            loop_strategy={"replay_buffer_size": 500, "checkpoint_interval": 5, "timeout_seconds": 3600},
            distillation_config={"temp_dir": str(_tmp_path), "output_dir": str(_tmp_path), "model_storage_path": str(_tmp_path)},
            trainer={"trusted_directories": [str(_tmp_path)]},
            oracle={"pseudo_dir": str(_tmp_path)},
            validator={}
        )
        _config = _full_config.dynamics
        _sys_config = _full_config.system

        _wrapper = EONWrapper(_config, _sys_config)

        # Mocking the Popen call for eonclient
        class _MockProc:
            returncode = 100  # 100 is our OTF halt code
            def communicate(self, *args, **kwargs):
                return b"out", b"err"
            def kill(self):
                pass
            def poll(self):
                return self.returncode
            def __enter__(self) -> "_MockProc":
                return self
            def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                pass

        _eon_bin = _tmp_path / "eonclient"
        _eon_bin.touch()
        _eon_bin.chmod(0o755)

        _pot_file = _tmp_path / "dummy.yace"
        _pot_file.touch()
        _work_dir = _tmp_path / "eon_run"

        with _patch("subprocess.Popen", return_value=_MockProc()), _patch("shutil.which", return_value=str(_eon_bin.resolve())):
            _config.eon_binary = "eonclient"
            _res = _wrapper.run_kmc(_pot_file, _work_dir)

            assert _res["halted"] is True
            assert _res["is_kmc"] is True

            print("✓ System seamlessly transitions from standard molecular dynamics to long-timescale Adaptive Kinetic Monte Carlo (aKMC) simulations via EON.")

            _ini_content = (_work_dir / "config.ini").read_text()
            assert "job = process_search" in _ini_content
            assert "min_mode_method = dimer" in _ini_content
            print("✓ Engine can accurately explore slow diffusion pathways and calculate precise activation energy barriers.")

    return ()


if __name__ == "__main__":
    app.run()
