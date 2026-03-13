from pathlib import Path
from unittest.mock import MagicMock, patch

from src.core.orchestrator import ActiveLearningOrchestrator
from src.domain_models.config import PipelineConfig


def test_orchestrator_get_latest_potential(tmp_path: Path) -> None:
    config = PipelineConfig()
    orchestrator = ActiveLearningOrchestrator(config)

    with patch("src.core.orchestrator.Path") as mock_path_cls:
        mock_path = MagicMock()
        mock_path_cls.return_value = mock_path

        # Scenario 1: Directory does not exist
        mock_path.exists.return_value = False
        assert orchestrator.get_latest_potential() is None

        # Scenario 2: Directory exists but no files
        mock_path.exists.return_value = True
        mock_path.glob.return_value = []
        assert orchestrator.get_latest_potential() is None

        # Scenario 3: Returns files
        mock_path.glob.return_value = [Path("generation_001.yace"), Path("generation_002.yace")]
        assert orchestrator.get_latest_potential() == Path("generation_002.yace")


def test_orchestrator_run_cycle_converged() -> None:
    config = PipelineConfig()
    orchestrator = ActiveLearningOrchestrator(config)

    with (
        patch.object(orchestrator, "get_latest_potential", return_value=None),
        patch.object(orchestrator.md_engine, "run_exploration") as mock_explore,
    ):
        mock_explore.return_value = {"halted": False}

        result = orchestrator.run_cycle()

        assert result == "CONVERGED"
        mock_explore.assert_called_once()


def test_orchestrator_run_cycle_error_dft() -> None:
    config = PipelineConfig()
    orchestrator = ActiveLearningOrchestrator(config)

    with (
        patch.object(orchestrator, "get_latest_potential", return_value=None),
        patch.object(orchestrator.md_engine, "run_exploration") as mock_explore,
        patch.object(orchestrator.md_engine, "extract_high_gamma_structures") as mock_extract,
        patch.object(orchestrator.trainer, "select_local_active_set") as mock_select,
        patch.object(orchestrator.oracle, "compute_batch") as mock_compute,
    ):
        mock_explore.return_value = {"halted": True, "dump_file": Path("dummy")}
        mock_extract.return_value = ["dummy_atom"]
        mock_select.return_value = ["dummy_candidate"]
        mock_compute.return_value = []  # Empty list = error

        result = orchestrator.run_cycle()

        assert result == "ERROR"


def test_orchestrator_run_cycle_validation_failed() -> None:
    config = PipelineConfig()
    orchestrator = ActiveLearningOrchestrator(config)

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
        mock_extract.return_value = ["dummy_atom"]
        mock_select.return_value = ["dummy_candidate"]
        mock_compute.return_value = ["dummy_result"]
        mock_update.return_value = Path("dummy_dataset")
        mock_train.return_value = Path("dummy_pot")
        mock_validate.return_value = {"passed": False, "reason": "Failed test"}

        result = orchestrator.run_cycle()

        assert result == "VALIDATION_FAILED"


def test_orchestrator_run_cycle_continue() -> None:
    config = PipelineConfig()
    orchestrator = ActiveLearningOrchestrator(config)

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
        mock_extract.return_value = ["dummy_atom"]
        mock_select.return_value = ["dummy_candidate"]
        mock_compute.return_value = ["dummy_result"]
        mock_update.return_value = Path("dummy_dataset")
        mock_train.return_value = Path("dummy_pot")
        mock_validate.return_value = {"passed": True}

        result = orchestrator.run_cycle()

        assert result == "CONTINUE"
        assert orchestrator.iteration == 2
        mock_copy.assert_called_once()
