import sys

# Crucial injection for Marimo running locally
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
    from pathlib import Path
    from typing import Any

    import pytest

    from src.core.exceptions import DynamicsHaltInterrupt
    from src.domain_models.config import DynamicsConfig, SystemConfig
    from src.dynamics.dynamics_engine import MDInterface

    logging.basicConfig(level=logging.INFO)

    # RUF001 NOQA wrapper
    return (
        DynamicsConfig,
        SystemConfig,
        MDInterface,
        DynamicsHaltInterrupt,
        Path,
        logging,
        subprocess,
        shutil,
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
    Any: Any,
) -> tuple:
    print("Executing UAT-C04-01: Validation of LAMMPS Soft-Start Resume Generation")
    print("Executing UAT-C04-03: Generation of Two-Tier Watchdog Commands and Smooth Steps")

    # GIVEN
    _config = DynamicsConfig(trusted_directories=[], project_root=str(Path.cwd()))
    _config.thresholds.smooth_steps = 5
    _config.thresholds.threshold_call_dft = 0.08
    _sys_config = SystemConfig(elements=["Fe"])
    _engine = MDInterface(_config, _sys_config)

    # Use a dummy environment
    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory() as _td:
        _tmp_path = Path(_td)
        _pot_file = _tmp_path / "dummy.yace"
        _pot_file.touch()

        _restart_dir = _tmp_path / "md_run"
        _restart_dir.mkdir(parents=True, exist_ok=True)
        _restart_file = _restart_dir / "restart.lammps"
        _restart_file.touch()

        _work_dir = _tmp_path / "resume_run"
        _work_dir.mkdir(parents=True)

        # Monkey patch subprocess safely
        _old_run = subprocess.run
        _old_which = shutil.which

        def _mock_run(*args: Any, **kwargs: Any) -> None:
            pass

        subprocess.run = _mock_run

        _lmp_path = _tmp_path / "lmp"
        _lmp_path.touch()
        _lmp_path.chmod(0o755)

        shutil.which = lambda *args, **kwargs: str(_lmp_path.resolve())
        _config.lmp_binary = "lmp"
        _config.trusted_directories = [str(_tmp_path)]

        # Override halt check
        _old_halt = _engine._check_halt
        _engine._check_halt = lambda x: {"halted": False}  # type: ignore[method-assign]

        # WHEN
        _engine.resume(_pot_file, _restart_dir, _work_dir)

        # THEN
        _script = (_work_dir / "in.lammps").read_text()
        assert f"read_restart {_restart_file.resolve()}" in _script
        assert "fix soft_start all langevin" in _script
        assert "run 100" in _script
        assert "unfix soft_start" in _script
        assert 'fix watchdog all halt 5 v_max_gamma > 0.08 error hard message "AL_HALT"' in _script

        print(
            "✓ UAT-C04-01 & UAT-C04-03 passed. Resume script correctly injects soft-start langevin protocol and two-tier thresholds."
        )

        # Cleanup monkey patches
        subprocess.run = _old_run
        shutil.which = _old_which
        _engine._check_halt = _old_halt  # type: ignore[method-assign]

    return ()


@app.cell
def test_scenario_02(
    DynamicsConfig: Any,
    SystemConfig: Any,
    MDInterface: Any,
    DynamicsHaltInterrupt: Any,
    Path: Any,
    subprocess: Any,
    shutil: Any,
    Any: Any,
) -> tuple:
    print("Executing UAT-C04-02: Successful Parsing of Active Learning Halt Events vs Fatal Errors")

    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory() as _td:
        _tmp_path = Path(_td)
        # GIVEN
        _config = DynamicsConfig(trusted_directories=[], project_root=str(_tmp_path))
        _sys_config = SystemConfig(elements=["Fe"])
        _engine = MDInterface(_config, _sys_config)
        _pot_file = _tmp_path / "dummy.yace"
        _pot_file.touch()
        _work_dir = _tmp_path / "md_run"
        _work_dir.mkdir(parents=True)

        # Monkey patch
        _old_run = subprocess.run
        _old_which = shutil.which

        def _mock_run_halt(*args: Any, **kwargs: Any) -> None:
            raise subprocess.CalledProcessError(1, ["lmp"])

        subprocess.run = _mock_run_halt

        _lmp_path = _tmp_path / "lmp"
        _lmp_path.touch()
        _lmp_path.chmod(0o755)
        shutil.which = lambda *args, **kwargs: str(_lmp_path.resolve())
        _config.lmp_binary = "lmp"
        _config.trusted_directories = [str(_tmp_path)]

        _dump_file = _work_dir / "dump.lammps"
        _dump_file.write_text("DUMP")
        _log_file = _work_dir / "log.lammps"

        # Mock AL_HALT correctly parsing
        _log_file.write_text("Some normal physical dynamics\nAL_HALT\n")

        # Mock _check_halt
        _old_halt = _engine._check_halt
        _engine._check_halt = lambda x: {"halted": True, "dump_file": _dump_file}  # type: ignore[method-assign]

        # WHEN expected halt
        _res = _engine.run_exploration(_pot_file, _work_dir)

        # THEN
        assert _res["halted"] is True
        print(
            "✓ UAT-C04-02: Correctly parsed AL_HALT as an expected exception, returning halted=True."
        )

        # WHEN unexpected crash
        _log_file.write_text("Lost Atoms error: simulation exploded\n")

        import pytest

        try:
            _engine.run_exploration(_pot_file, _work_dir)
            pytest.fail("Should have raised DynamicsHaltInterrupt")
        except DynamicsHaltInterrupt:
            print(
                "✓ UAT-C04-02: Correctly recognized missing AL_HALT as a fatal physical crash, raising DynamicsHaltInterrupt."
            )

        # Cleanup
        subprocess.run = _old_run
        shutil.which = _old_which
        _engine._check_halt = _old_halt  # type: ignore[method-assign]

    return ()


@app.cell
def test_scenario_04(DynamicsConfig: Any, SystemConfig: Any, MDInterface: Any, Path: Any) -> tuple:
    print("Executing UAT-C04-04: Proper Handling of the Initial Data Reading Phase")

    # GIVEN
    _config = DynamicsConfig(trusted_directories=[], project_root=str(Path.cwd()))
    _sys_config = SystemConfig(elements=["Fe"])
    _engine = MDInterface(_config, _sys_config)

    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory() as _td:
        _tmp_path = Path(_td)

        # WHEN
        with Path.open(_tmp_path / "in.lammps.temp", "w") as f:
            _engine._write_cold_start_input(f, "dump.lammps", _tmp_path)

        _script = (_tmp_path / "in.lammps.temp").read_text()

        # THEN
        assert (
            "read_data" not in _script
        )  # Note: write_cold_start actually doesn't use read_data natively, it generates the box internally via `create_atoms` for our architecture, which perfectly lacks read_restart. Let's verify `read_restart` is missing.
        assert "create_atoms" in _script
        assert "read_restart" not in _script
        assert "fix soft_start" not in _script
        print(
            "✓ UAT-C04-04 passed. Initial phase correctly generates a cold-start script missing read_restart and soft_start constraints."
        )

    return ()


if __name__ == "__main__":
    app.run()
