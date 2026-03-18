import marimo

__generated_with = "0.10.19"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path.cwd()))
    import logging
    import shutil
    import sqlite3
    from pathlib import Path
    from unittest.mock import MagicMock

    from src.core.exceptions import DynamicsHaltInterrupt
    from src.core.orchestrator import Orchestrator

    # Supress noisy logs
    logging.getLogger().setLevel(logging.ERROR)

    # Mock heavy dependencies
    sys.modules["pyacemaker"] = MagicMock()
    sys.modules["pyacemaker.calculator"] = MagicMock()

    # Setup test workspace
    tmp_path = Path("/tmp/uat_c06_workspace")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True)

    # Create valid dummy env to load config
    env_file = tmp_path / ".env"
    env_file.write_text("MLIP_DYNAMICS__UNCERTAINTY_THRESHOLD=5.0\n")

    import typing

    # Instead of fully initializing config and dealing with dotenv validation which expects root .env,
    # we'll mock the config entirely.
    class DummyConfig:
        class LoopStrategy:
            use_tiered_oracle: typing.ClassVar[bool] = False
            max_iterations: typing.ClassVar[int] = 2  # Enough to run full 4-phase and loop

            class Thresholds:
                threshold_call_dft: typing.ClassVar[float] = 0.5

            thresholds = Thresholds()

        loop_strategy = LoopStrategy()

        class System:
            elements: typing.ClassVar[list[str]] = ["Fe"]
            interface_target: typing.ClassVar[str | None] = None
            interface_generation_iteration: typing.ClassVar[int] = 0
            restricted_directories: typing.ClassVar[list[str]] = []
            baseline_potential: typing.ClassVar[str] = "zbl"

        system = System()

        class Dynamics:
            trusted_directories: typing.ClassVar[list[str]] = []
            project_root: typing.ClassVar[str] = str(tmp_path)

        dynamics = Dynamics()

        class Oracle:
            pass

        oracle = Oracle()

        class Trainer:
            trusted_directories: typing.ClassVar[list[str]] = []
            max_potential_size: typing.ClassVar[int] = 1000000

        trainer = Trainer()

        class Validator:
            pass

        validator = Validator()

        class StructureGenerator:
            pass

        structure_generator = StructureGenerator()

        class Policy:
            pass

        policy = Policy()

        project_root = tmp_path

    return DummyConfig, Orchestrator, tmp_path, DynamicsHaltInterrupt, sqlite3


@app.cell
def test_scenario_1(DummyConfig, Orchestrator, tmp_path):
    import tempfile

    # Scenario ID: UAT-C06-01: HPC Wall-Time Job Kill Recovery and State Resumption
    print("Testing UAT-C06-01: HPC Kill Recovery")

    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path

        temp_dir = Path(td).resolve()
        config = DummyConfig(str(temp_dir))

        orch1 = Orchestrator(config)
        # Simulate a run that completed Phase1 and is killed
        orch1.checkpoint.set_state("CURRENT_PHASE", "PHASE2_VALIDATION")
        orch1.checkpoint.set_state("CURRENT_ITERATION", 5)

        # "Restart" the process
        orch2 = Orchestrator(config)

        assert orch2.checkpoint.get_state("CURRENT_PHASE") == "PHASE2_VALIDATION", (
            "State did not persist!"
        )
        assert orch2.iteration == 5, (
            f"Iteration counter did not resume correctly! It is {orch2.iteration}"
        )
        print("✓ Orchestrator successfully resumed from database checkpoint")
        return orch1, orch2


@app.cell
def test_scenario_2(DummyConfig, Orchestrator, tmp_path, DynamicsHaltInterrupt):
    # Scenario ID: UAT-C06-02: Execution of the Full 4-Phase Loop
    print("\nTesting UAT-C06-02: 4-Phase Hierarchical Workflow")
    import unittest.mock

    _MagicMock = unittest.mock.MagicMock

    _orch = Orchestrator(DummyConfig())
    # Clean state
    _orch.checkpoint.set_state("CURRENT_PHASE", "PHASE1_DISTILLATION")
    _orch.iteration = 0

    # Setup mocks
    _orch.get_latest_potential = _MagicMock(return_value=tmp_path / "pot.yace")
    _orch._pre_generate_interface_target = _MagicMock(return_value=None)
    _orch._validate_potential = _MagicMock(return_value=True)
    _orch._deploy_potential = _MagicMock(return_value=tmp_path / "new.yace")
    _orch._finalize_directories = _MagicMock()
    _orch._resume_md_engine = _MagicMock()
    _orch._run_dft_and_train = _MagicMock(return_value=tmp_path / "new.yace")
    _orch._select_candidates = _MagicMock(return_value=iter([]))

    # 1st loop: Halt in MD
    # 2nd loop: Completes normally and iteration increments
    side_effects = [DynamicsHaltInterrupt("Simulated Halt"), None]
    _orch._run_exploration = _MagicMock(side_effect=side_effects)

    _orch.run_cycle()

    assert _orch.iteration == 2, "Did not complete expected iterations"
    assert _orch._run_exploration.call_count == 2, "Exploration not called correctly"
    assert _orch._run_dft_and_train.call_count == 1, "DFT not called on halt"
    print("✓ Orchestrator successfully handled Halt -> DFT -> Finetune -> Resume -> Next Loop")
    return _orch


@app.cell
def test_scenario_3(DummyConfig, Orchestrator, tmp_path):
    # Scenario ID: UAT-C06-03: Automated Artifact Cleanup
    print("\nTesting UAT-C06-03: Artifact Cleanup Daemon")
    _orch2 = Orchestrator(DummyConfig())

    huge_file = tmp_path / "wfc.dat"
    huge_file.write_bytes(b"0" * 1024)

    assert huge_file.exists()
    _orch2._cleanup_artifacts([huge_file])
    assert not huge_file.exists(), "Cleanup daemon failed to delete artifact!"
    print("✓ Cleanup daemon aggressively deleted massive artifacts")
    return huge_file


@app.cell
def test_scenario_4(DummyConfig, Orchestrator, tmp_path, sqlite3):
    # Scenario ID: UAT-C06-04: Handling Corrupted State Files
    print("\nTesting UAT-C06-04: Handling Corrupted Checkpoint DB")

    db_path = tmp_path / ".ac_cdd" / "checkpoint.db"

    # Make DB read only to simulate lock/permission error
    from pathlib import Path

    Path(db_path).chmod(0o400)

    try:
        _orch3 = Orchestrator(DummyConfig())
        _orch3.checkpoint.set_state("TEST", "123")
        failed = False
    except RuntimeError as e:
        failed = True
        if "Failed to set state" not in str(e):
            msg = "Did not raise expected error message"
            raise AssertionError(msg) from e

    # restore permission so it can be deleted later
    from pathlib import Path

    Path(db_path).chmod(0o600)

    if not failed:
        msg = "Orchestrator did not fail loudly on corrupted DB"
        raise AssertionError(msg)
    print("✓ Orchestrator successfully failed loudly without overwriting locked DB")
    return db_path


if __name__ == "__main__":
    app.run()
