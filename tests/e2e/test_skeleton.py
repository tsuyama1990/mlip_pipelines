from pathlib import Path
from typing import Any

from src.core.orchestrator import Orchestrator
from src.domain_models.config import (
    DynamicsConfig,
    OracleConfig,
    ProjectConfig,
    SystemConfig,
    TrainerConfig,
    ValidatorConfig,
)


def test_full_pipeline_skeleton(tmp_path: Path) -> None:  # noqa: C901
    config = ProjectConfig(
        system=SystemConfig(elements=["Fe", "Pt"], baseline_potential="zbl"),
        dynamics=DynamicsConfig(uncertainty_threshold=5.0, md_steps=100),
        oracle=OracleConfig(kspacing=0.1, smearing_width=0.02, pseudo_dir=str(tmp_path)),
        trainer=TrainerConfig(max_epochs=2, active_set_size=10),
        validator=ValidatorConfig(energy_rmse_threshold=0.05),
        project_root=tmp_path,
    )

    orchestrator = Orchestrator(config)
    assert orchestrator.iteration == 0

    # Run cycle directly
    # To avoid hanging on real DFT or lammps in tests, we patch internal calls
    class MockMD:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def run_exploration(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"halted": True, "dump_file": tmp_path / "dummy_dump"}

        def extract_high_gamma_structures(self, *args: Any, **kwargs: Any) -> list[Any]:
            from ase import Atoms

            return [Atoms("Fe", positions=[(0, 0, 0)])]

    class MockOracle:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def compute_batch(self, batch: Any, *args: Any, **kwargs: Any) -> Any:
            # return the same batch to pretend we labeled them
            return batch

    class MockTrainer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def select_local_active_set(self, candidates: list[Any], anchor: Any, n: int) -> list[Any]:
            return candidates[:n]

        def update_dataset(self, new_data: Any, dataset_path: Any) -> Any:
            return dataset_path

        def train(self, dataset: Any, initial_potential: Any, output_dir: Path) -> Path:
            pot = output_dir / "output_potential.yace"
            pot.parent.mkdir(parents=True, exist_ok=True)
            pot.write_text("dummy potential")
            return pot

    class MockValidator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def validate(self, *args: Any, **kwargs: Any) -> Any:
            from src.domain_models.dtos import ValidationReport

            return ValidationReport(
                passed=True,
                energy_rmse=0.001,
                force_rmse=0.01,
                stress_rmse=0.05,
                phonon_stable=True,
                mechanically_stable=True,
            )

    orchestrator.md_engine = MockMD()  # type: ignore[assignment]
    orchestrator.oracle = MockOracle()  # type: ignore[assignment]
    orchestrator.trainer = MockTrainer()  # type: ignore[assignment]
    orchestrator.validator = MockValidator()  # type: ignore[assignment]

    result = orchestrator.run_cycle()
    assert result is not None
    assert orchestrator.iteration == 1
    assert (tmp_path / "potentials" / "generation_001.yace").exists()
