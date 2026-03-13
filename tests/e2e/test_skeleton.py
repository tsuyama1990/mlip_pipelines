from src.core.orchestrator import Orchestrator
from src.domain_models.config import (
    DynamicsConfig,
    OracleConfig,
    ProjectConfig,
    SystemConfig,
    TrainerConfig,
    ValidatorConfig,
)


def test_full_pipeline_skeleton(tmp_path):
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
        def __init__(self, *args, **kwargs):
            pass

        def run_exploration(self, *args, **kwargs):
            return {"halted": True, "dump_file": tmp_path / "dummy_dump"}

        def extract_high_gamma_structures(self, *args, **kwargs):
            from ase import Atoms

            return [Atoms("Fe", positions=[(0, 0, 0)])]

    class MockOracle:
        def __init__(self, *args, **kwargs):
            pass

        def compute_batch(self, batch, *args, **kwargs):
            # return the same batch to pretend we labeled them
            return batch

    class MockTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def select_local_active_set(self, candidates, anchor, n):
            return candidates[:n]

        def update_dataset(self, new_data, dataset_path):
            return dataset_path

        def train(self, dataset, initial_potential, output_dir):
            pot = output_dir / "new_pot.yace"
            pot.parent.mkdir(parents=True, exist_ok=True)
            pot.write_text("dummy potential")
            return pot

    class MockValidator:
        def __init__(self, *args, **kwargs):
            pass

        def validate(self, *args, **kwargs):
            from src.domain_models.dtos import ValidationReport

            return ValidationReport(
                passed=True,
                energy_rmse=0.001,
                force_rmse=0.01,
                stress_rmse=0.05,
                phonon_stable=True,
                mechanically_stable=True,
            )

    orchestrator.md_engine = MockMD()
    orchestrator.oracle = MockOracle()
    orchestrator.trainer = MockTrainer()
    orchestrator.validator = MockValidator()

    result = orchestrator.run_cycle()
    assert result is not None
    assert orchestrator.iteration == 1
    assert (tmp_path / "potentials" / "generation_001.yace").exists()
