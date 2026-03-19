import marimo

__generated_with = "0.1.0"
app = marimo.App()


@app.cell
def __1():  # type: ignore[no-untyped-def]

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path.cwd()))
    return sys, Path


@app.cell
def __2(sys, Path):  # type: ignore[no-untyped-def]

    import tempfile

    from src.domain_models.config import (
        ActiveLearningThresholds,
        CutoutConfig,
        DistillationConfig,
        DynamicsConfig,
        LoopStrategyConfig,
        OracleConfig,
        ProjectConfig,
        SystemConfig,
        TrainerConfig,
        ValidatorConfig,
    )

    tmp_dir = Path(tempfile.mkdtemp())
    (tmp_dir / "pyproject.toml").touch()

    config = ProjectConfig(
        project_root=tmp_dir,
        system=SystemConfig(elements=["Fe", "Pt", "Mg", "O"]),
        dynamics=DynamicsConfig(
            project_root=str(tmp_dir), thresholds=ActiveLearningThresholds(), trusted_directories=[]
        ),
        oracle=OracleConfig(),
        trainer=TrainerConfig(trusted_directories=[]),
        validator=ValidatorConfig(),
        distillation_config=DistillationConfig(
            temp_dir=str(tmp_dir), output_dir=str(tmp_dir), model_storage_path=str(tmp_dir)
        ),
        loop_strategy=LoopStrategyConfig(
            replay_buffer_size=100, checkpoint_interval=10, max_retries=3, timeout_seconds=3600
        ),
        cutout_config=CutoutConfig(),
    )
    print("Config successfully loaded!")
    return config, tmp_dir


@app.cell
def __3(config, sys, tmp_dir):  # type: ignore[no-untyped-def]

    import importlib.util

    # Hybrid Environment Support: Mock pyacemaker and phonopy if they do not exist
    # Mocks must be injected BEFORE importing Orchestrator to prevent ModuleNotFoundError
    if importlib.util.find_spec("pyacemaker") is None:
        print("Skipping step (No pyacemaker available) - using mock mode for pyacemaker.")
        import types
        pyacemaker_mock = types.ModuleType("pyacemaker")
        pyacemaker_calculator_mock = types.ModuleType("pyacemaker.calculator")

        def mock_pyacemaker(*args, **kwargs):  # type: ignore[no-untyped-def]
            from ase.calculators.lj import LennardJones
            return LennardJones()

        pyacemaker_calculator_mock.pyacemaker = mock_pyacemaker
        sys.modules["pyacemaker"] = pyacemaker_mock
        sys.modules["pyacemaker.calculator"] = pyacemaker_calculator_mock

    if importlib.util.find_spec("phonopy") is None:
        print("Skipping step (No phonopy available) - using mock mode for phonopy.")
        import types
        phonopy_mock = types.ModuleType("phonopy")
        sys.modules["phonopy"] = phonopy_mock

    from src.core.orchestrator import Orchestrator

    # Force the policy engine to always return a Fallback Standard strategy (MD)
    # by raising a handled error when decide_policy is called.
    orchestrator = Orchestrator(config)

    # Mock the decide_policy to return Fallback Standard which uses md_mc_ratio=0.0
    # to avoid EON kMC execution which requires the external eonclient.
    import unittest.mock

    from src.domain_models.dtos import ExplorationStrategy
    orchestrator.policy_engine.decide_policy = unittest.mock.MagicMock(
        return_value=ExplorationStrategy(
            md_mc_ratio=0.0,
            t_max=300.0,
            n_defects=0.0,
            strain_range=0.0,
            policy_name="Fallback Standard"
        )
    )
    print("Orchestrator successfully initialized!")
    return orchestrator,


@app.cell
def __4(orchestrator):  # type: ignore[no-untyped-def]
    # Quick Start Guide
    print("--- Quick Start Guide ---")
    print(f"Project initialized at {orchestrator.config.project_root}")
    # Simulating a single active learning cycle in mock mode

    # In mock mode, we expect certain exceptions because underlying binaries might be missing.
    # However, we only catch those specific mocked/expected exceptions to prevent swallowing actual regressions!
    try:
        res = orchestrator.run_cycle()
        print(f"Cycle completed with result: {res}")
    except RuntimeError as e:
        if "No valid data obtained from DFT" in str(e):
            print(f"Cycle completed with simulated execution (Mock Exception gracefully handled): {e}")
        else:
            raise

    print("Tutorial execution successfully completed.")
    return ()


@app.cell
def __5(config):  # type: ignore[no-untyped-def]
    # Advanced Use Case
    print("--- Advanced Use Case: Modifying Thresholds ---")
    config.loop_strategy.thresholds.threshold_call_dft = 0.5
    config.loop_strategy.thresholds.threshold_add_train = 0.8
    print(f"Updated DFT call threshold: {config.loop_strategy.thresholds.threshold_call_dft}")
    print(f"Updated add to train threshold: {config.loop_strategy.thresholds.threshold_add_train}")
    return ()


if __name__ == "__main__":
    app.run()
