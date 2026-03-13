from pathlib import Path
from unittest.mock import patch

from src.core.orchestrator import ActiveLearningOrchestrator
from src.domain_models.config import PipelineConfig
from src.dynamics.dynamics_engine import DynamicsEngine
from src.generators.adaptive_policy import AdaptivePolicy
from src.oracles.dft_oracle import DFTOracle
from src.trainers.ace_trainer import ACETrainer
from src.validators.validator import Validator


def _create_test_orchestrator(config: PipelineConfig) -> ActiveLearningOrchestrator:
    return ActiveLearningOrchestrator(
        config=config,
        md_engine=DynamicsEngine(config.lammps, config.otf_loop, config.material),
        oracle=DFTOracle(config.dft),
        trainer=ACETrainer(config.training),
        validator=Validator(config.validation, config.material),
        policy_engine=AdaptivePolicy(
            {"elements": config.material.elements},
            {
                "band_gap": config.material.band_gap,
                "melting_point": config.material.melting_point,
                "bulk_modulus": config.material.bulk_modulus,
            },
            config.policy,
        ),
    )


def test_orchestrator_get_latest_potential(
    mock_pipeline_config: PipelineConfig, tmp_path: Path
) -> None:
    # Modify config to point to our tmp_path
    mock_pipeline_config.potential_path_template = str(
        tmp_path / "potentials" / "generation_{iteration:03d}.yace"
    )
    config = mock_pipeline_config
    orchestrator = _create_test_orchestrator(config)
    pot_dir = tmp_path / "potentials"

    # Scenario 1: Directory does not exist
    assert orchestrator.get_latest_potential() is None

    # Scenario 2: Directory exists but no files
    pot_dir.mkdir()
    assert orchestrator.get_latest_potential() is None

    # Scenario 3: Returns files
    (pot_dir / "generation_001.yace").touch()
    (pot_dir / "generation_002.yace").touch()

    latest = orchestrator.get_latest_potential()
    assert latest is not None
    assert latest.name == "generation_002.yace"


def test_orchestrator_run_cycle_converged(mock_pipeline_config: PipelineConfig) -> None:
    config = mock_pipeline_config
    orchestrator = _create_test_orchestrator(config)

    with (
        patch.object(orchestrator, "get_latest_potential", return_value=None),
        patch.object(orchestrator.md_engine, "run_exploration") as mock_explore,
    ):
        mock_explore.return_value = {"halted": False}

        result = orchestrator.run_cycle()

        assert result == "CONVERGED"
        mock_explore.assert_called_once()


def test_orchestrator_run_cycle_error_dft(mock_pipeline_config: PipelineConfig) -> None:
    config = mock_pipeline_config
    orchestrator = _create_test_orchestrator(config)

    with (
        patch.object(orchestrator, "get_latest_potential", return_value=None),
        patch.object(orchestrator.md_engine, "run_exploration") as mock_explore,
        patch.object(orchestrator.md_engine, "extract_high_gamma_structures") as mock_extract,
        patch.object(orchestrator.trainer, "select_local_active_set") as mock_select,
        patch.object(orchestrator.oracle, "compute_batch") as mock_compute,
    ):
        mock_explore.return_value = {"halted": True, "dump_file": Path("dummy")}
        from ase.build import bulk

        mock_extract.return_value = [bulk("Fe", cubic=True)]
        mock_select.return_value = [bulk("Pt", cubic=True)]
        mock_compute.return_value = []  # Empty list = error

        result = orchestrator.run_cycle()

        assert result == "ERROR"


def test_orchestrator_run_cycle_validation_failed(mock_pipeline_config: PipelineConfig) -> None:
    config = mock_pipeline_config
    orchestrator = _create_test_orchestrator(config)

    with (
        patch.object(orchestrator, "get_latest_potential", return_value=None),
        patch.object(orchestrator.md_engine, "run_exploration") as mock_explore,
        patch.object(orchestrator.md_engine, "extract_high_gamma_structures") as mock_extract,
        patch.object(orchestrator.trainer, "select_local_active_set") as mock_select,
        patch.object(orchestrator.oracle, "compute_batch") as mock_compute,
        patch.object(orchestrator.trainer, "update_dataset") as mock_update,
        patch.object(orchestrator.trainer, "train") as mock_train,
        patch.object(orchestrator.validator, "validate") as mock_validate,
    ):
        mock_explore.return_value = {"halted": True, "dump_file": Path("dummy")}
        from ase.build import bulk

        mock_extract.return_value = [bulk("Fe", cubic=True)]
        mock_select.return_value = [bulk("Pt", cubic=True)]
        mock_compute.return_value = ["dummy_result"]
        mock_update.return_value = Path("dummy_dataset")
        mock_train.return_value = Path("dummy_pot")
        mock_validate.return_value = {"passed": False, "reason": "Failed test"}

        result = orchestrator.run_cycle()

        assert result == "VALIDATION_FAILED"


def test_orchestrator_run_cycle_continue(mock_pipeline_config: PipelineConfig) -> None:
    config = mock_pipeline_config
    orchestrator = _create_test_orchestrator(config)

    with (
        patch.object(orchestrator, "get_latest_potential", return_value=None),
        patch.object(orchestrator.md_engine, "run_exploration") as mock_explore,
        patch.object(orchestrator.md_engine, "extract_high_gamma_structures") as mock_extract,
        patch.object(orchestrator.trainer, "select_local_active_set") as mock_select,
        patch.object(orchestrator.oracle, "compute_batch") as mock_compute,
        patch.object(orchestrator.trainer, "update_dataset") as mock_update,
        patch.object(orchestrator.trainer, "train") as mock_train,
        patch.object(orchestrator.validator, "validate") as mock_validate,
        patch("src.core.orchestrator.shutil.copy") as mock_copy,
    ):
        mock_explore.return_value = {"halted": True, "dump_file": Path("dummy")}
        from ase.build import bulk

        mock_extract.return_value = [bulk("Fe", cubic=True)]
        mock_select.return_value = [bulk("Pt", cubic=True)]
        mock_compute.return_value = ["dummy_result"]
        mock_update.return_value = Path("dummy_dataset")
        mock_train.return_value = Path("dummy_pot")
        mock_validate.return_value = {"passed": True}

        result = orchestrator.run_cycle()

        assert result == "CONTINUE"
        assert orchestrator.iteration == 2
        mock_copy.assert_called_once()
