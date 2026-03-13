from src.core.orchestrator import Orchestrator
from src.domain_models.config import ProjectConfig


def test_orchestrator_initialization(mock_project_config: ProjectConfig):
    orch = Orchestrator(mock_project_config)
    assert orch.config.system.elements == ["Fe", "Pt"]
    assert orch.iteration == 0

def test_run_cycle(monkeypatch, mock_project_config: ProjectConfig):
    orch = Orchestrator(mock_project_config)

    # Mock all internal models
    class MockMD:
        def run_exploration(self, *args, **kwargs):
            return {"halted": True, "dump_file": "dummy_dump"}
        def extract_high_gamma_structures(self, *args, **kwargs):
            from ase import Atoms
            return [Atoms("Fe", positions=[(0, 0, 0)])]

    class MockOracle:
        def compute_batch(self, *args, **kwargs):
            from ase import Atoms
            return [Atoms("Fe", positions=[(0, 0, 0)])]

    class MockTrainer:
        def get_latest_potential(self):
            return "dummy_pot.yace"
        def select_local_active_set(self, *args, **kwargs):
            from ase import Atoms
            return [Atoms("Fe", positions=[(0, 0, 0)])]*5
        def update_dataset(self, *args, **kwargs):
            from pathlib import Path
            return Path("dummy.pckl")
        def train(self, dataset, initial_potential, output_dir):
            pot = output_dir / "new_pot.yace"
            pot.parent.mkdir(parents=True, exist_ok=True)
            pot.write_text("dummy potential")
            return pot

    class MockValidator:
        def validate(self, *args, **kwargs):
            from src.domain_models.dtos import ValidationReport
            return ValidationReport(passed=True, energy_rmse=0.001, force_rmse=0.01, stress_rmse=0.05, phonon_stable=True, mechanically_stable=True)

    orch.md_engine = MockMD()
    orch.oracle = MockOracle()
    orch.trainer = MockTrainer()
    orch.validator = MockValidator()

    # Mock structure generator
    class MockGenerator:
        def generate_local_candidates(self, s0, n=20):
            from ase import Atoms
            return [Atoms("Fe", positions=[(0, 0, 0)])]*n

    orch.structure_generator = MockGenerator()

    res = orch.run_cycle()
    assert orch.iteration == 1
    assert res is not None
    assert str(res).endswith("generation_001.yace")
