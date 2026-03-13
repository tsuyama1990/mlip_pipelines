import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.core.orchestrator import ActiveLearningOrchestrator
from src.domain_models.config import PipelineConfig


@patch.dict(sys.modules, {"lammps": MagicMock()})
def test_pipeline_skeleton(tmp_path: Path) -> None:
    """Verifies that the orchestrator goes through the halt and heal cycle."""
    config = PipelineConfig()

    orchestrator = ActiveLearningOrchestrator(config)

    with (
        patch("src.oracles.dft_oracle.Espresso"),
        patch("subprocess.run"),
        patch("shutil.which", return_value="pace_train"),
        patch("src.validators.validator.Validator._check_phonons", return_value=True),
        patch("ase.Atoms.get_potential_energy", return_value=-100.0),
        patch("ase.Atoms.get_forces", return_value=__import__("numpy").zeros((2, 3))),
    ):
        # For Lammps halt loop
        mock_lammps = sys.modules["lammps"]
        lmp_instance = MagicMock()
        mock_lammps.lammps.return_value = lmp_instance

        # Throw exception on 'run' command to simulate error hard
        def mock_command(cmd: str) -> None:
            if "run" in cmd:
                msg = "LAMMPS error hard triggered by watchdog"
                raise RuntimeError(msg)

        lmp_instance.command.side_effect = mock_command
        lmp_instance.extract_variable.return_value = 6.0

        # Run one cycle
        result = orchestrator.run_cycle()

        # Can be CONVERGED or CONTINUE. It should not be ERROR or VALIDATION_FAILED.
        assert result in ["CONVERGED", "CONTINUE"]

        # If it continued, it should have created the outputs.
        if result == "CONTINUE":
            assert (Path("potentials") / "generation_001.yace").exists()
            assert (Path("active_learning") / "iter_001").exists()
