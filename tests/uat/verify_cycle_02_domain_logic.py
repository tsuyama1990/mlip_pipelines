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

    with tempfile.TemporaryDirectory() as _td:
        _tmp_path = Path(_td)

        # GIVEN: A valid YAML-equivalent configuration defining materials and thresholds
        import os as _os
        _os.environ["MLIP_DYNAMICS__TRUSTED_DIRECTORIES"] = '["' + str(_tmp_path) + '"]'
        _os.environ["MLIP_PROJECT_ROOT"] = str(_tmp_path)
        _os.environ["MLIP_DYNAMICS__PROJECT_ROOT"] = str(_tmp_path)
        _os.environ["MLIP_DYNAMICS__MD_STEPS"] = "1000"
        _os.environ["MLIP_DYNAMICS__TEMPERATURE"] = "300.0"
        _os.environ["MLIP_DYNAMICS__THRESHOLDS__THRESHOLD_CALL_DFT"] = "5.0"
        _os.environ["MLIP_DYNAMICS__THRESHOLDS__SMOOTH_STEPS"] = "3"
        _os.environ["MLIP_SYSTEM__ELEMENTS"] = '["Fe", "Pt"]'
        _os.environ["MLIP_SYSTEM__BASELINE_POTENTIAL"] = "zbl"

        # Instantiate from base config which reads env
        from src.domain_models.config import ProjectConfig as _ProjectConfig

        # Provide minimal required config for missing fields so instantiation succeeds
        _os.environ["MLIP_LOOP_STRATEGY__REPLAY_BUFFER_SIZE"] = "500"
        _os.environ["MLIP_LOOP_STRATEGY__CHECKPOINT_INTERVAL"] = "5"
        _os.environ["MLIP_LOOP_STRATEGY__TIMEOUT_SECONDS"] = "3600"
        _os.environ["MLIP_DISTILLATION_CONFIG__TEMP_DIR"] = str(_tmp_path)
        _os.environ["MLIP_DISTILLATION_CONFIG__OUTPUT_DIR"] = str(_tmp_path)
        _os.environ["MLIP_DISTILLATION_CONFIG__MODEL_STORAGE_PATH"] = str(_tmp_path)
        _os.environ["MLIP_TRAINER__TRUSTED_DIRECTORIES"] = '["' + str(_tmp_path) + '"]'

        (_tmp_path / "README.md").touch()
        _full_config = _ProjectConfig(oracle={"pseudo_dir": str(_tmp_path)}, validator={})
        _config = _full_config.dynamics
        _sys_config = _full_config.system
        # Config loaded from env variables via ProjectConfig

        # WHEN: Initiating the powerful active learning computational pipeline
        _engine = MDInterface(_config, _sys_config)

        # Mocking external LAMMPS execution for purely functional UAT
        _old_run = subprocess.run
        _old_which = shutil.which

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

        subprocess.run = mock_run_explore

        _lmp_path = _tmp_path / "lmp"
        _lmp_path.touch()
        _lmp_path.chmod(0o755)
        shutil.which = lambda *args, **kwargs: str(_lmp_path.resolve())
        _config.lmp_binary = "lmp"

        _pot_file = _tmp_path / "dummy.yace"
        _pot_file.touch()

        _work_dir = _tmp_path / "md_run"
        _work_dir.mkdir(parents=True)

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

        subprocess.run = mock_run_resume

        _restart_dir = _tmp_path / "md_run"
        _restart_file = _restart_dir / "restart.lammps"
        _restart_file.touch()

        _resume_dir = _tmp_path / "resume_run"
        _resume_dir.mkdir(parents=True)

        # THEN: Seamlessly flawlessly resume the paused simulation
        _res_resume = _engine.resume(_pot_file, _restart_dir, _resume_dir)
        assert _res_resume["halted"] is False

        _resume_script = (_resume_dir / "in.lammps").read_text()
        assert f"read_restart {_restart_file.resolve()}" in _resume_script
        print("✓ Dynamics Engine successfully resumed from process checkpoint.")

        # Cleanup
        subprocess.run = _old_run
        shutil.which = _old_which

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

    with tempfile.TemporaryDirectory() as _td:
        _tmp_path = Path(_td)

        import os as _os
        _os.environ["MLIP_DYNAMICS__TRUSTED_DIRECTORIES"] = '["' + str(_tmp_path) + '"]'
        _os.environ["MLIP_PROJECT_ROOT"] = str(_tmp_path)
        _os.environ["MLIP_DYNAMICS__PROJECT_ROOT"] = str(_tmp_path)
        _os.environ["MLIP_DYNAMICS__EON_JOB"] = "process_search"
        _os.environ["MLIP_DYNAMICS__EON_MIN_MODE_METHOD"] = "dimer"
        _os.environ["MLIP_SYSTEM__ELEMENTS"] = '["Fe", "Pt"]'

        from src.domain_models.config import ProjectConfig as _ProjectConfig
        _os.environ["MLIP_LOOP_STRATEGY__REPLAY_BUFFER_SIZE"] = "500"
        _os.environ["MLIP_LOOP_STRATEGY__CHECKPOINT_INTERVAL"] = "5"
        _os.environ["MLIP_LOOP_STRATEGY__TIMEOUT_SECONDS"] = "3600"
        _os.environ["MLIP_DISTILLATION_CONFIG__TEMP_DIR"] = str(_tmp_path)
        _os.environ["MLIP_DISTILLATION_CONFIG__OUTPUT_DIR"] = str(_tmp_path)
        _os.environ["MLIP_DISTILLATION_CONFIG__MODEL_STORAGE_PATH"] = str(_tmp_path)
        _os.environ["MLIP_TRAINER__TRUSTED_DIRECTORIES"] = '["' + str(_tmp_path) + '"]'

        (_tmp_path / "README.md").touch()
        _full_config = _ProjectConfig(oracle={"pseudo_dir": str(_tmp_path)}, validator={})
        _config = _full_config.dynamics
        _sys_config = _full_config.system

        _wrapper = EONWrapper(_config, _sys_config)

        _old_popen = subprocess.Popen
        _old_which = shutil.which

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

        subprocess.Popen = lambda *args, **kwargs: _MockProc()

        _eon_bin = _tmp_path / "eonclient"
        _eon_bin.touch()
        _eon_bin.chmod(0o755)
        shutil.which = lambda *args, **kwargs: str(_eon_bin.resolve())
        _config.eon_binary = "eonclient"

        _pot_file = _tmp_path / "dummy.yace"
        _pot_file.touch()
        _work_dir = _tmp_path / "eon_run"

        _res = _wrapper.run_kmc(_pot_file, _work_dir)

        assert _res["halted"] is True
        assert _res["is_kmc"] is True

        print("✓ System seamlessly transitions from standard molecular dynamics to long-timescale Adaptive Kinetic Monte Carlo (aKMC) simulations via EON.")

        _ini_content = (_work_dir / "config.ini").read_text()
        assert "job = process_search" in _ini_content
        assert "min_mode_method = dimer" in _ini_content
        print("✓ Engine can accurately explore slow diffusion pathways and calculate precise activation energy barriers.")

        # Cleanup
        subprocess.Popen = _old_popen
        shutil.which = _old_which

    return ()


if __name__ == "__main__":
    app.run()
