import shutil
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from omegaconf import OmegaConf

# We need a potential implementation. For the test we can mock it or use a dummy.
# But the orchestrator expects AbstractPotential.
from core.interfaces import AbstractPotential
from core.orchestrator import ActiveLearningOrchestrator
from oracles.qe_oracle import QeOracle


# Mock Generator
class MockGenerator:
    def generate_initial_pool(self, n=5):
        # Return simple Si dimer in a box
        seeds = []
        for _ in range(n):
            atoms = Atoms('Si2', positions=[[0, 0, 0], [2.35, 0, 0]], cell=[10, 10, 10], pbc=True)
            seeds.append(atoms)
        return seeds

# Mock Potential
class MockPotential(AbstractPotential):
    def __init__(self):
        self.u_max = 1.0

    def train(self, training_data, **kwargs):
        # Update u_max or state
        self.u_max += 0.1

    def predict(self, atoms):
        # Return dummy values
        return 0.0, np.zeros((len(atoms), 3)), np.zeros((3, 3))

    def get_uncertainty(self, atoms):
        # Return random uncertainty around 1.0
        return np.random.uniform(0.8, 1.5, size=len(atoms))

    def save(self, path):
        # touch file
        Path(path).touch()

    def load(self, path):
        pass

    # ASE Calculator compatibility
    def get_potential_energy(self, atoms=None, force_consistent=False):
        return 0.0

    def get_forces(self, atoms=None):
        return np.zeros((len(atoms), 3))

    def get_stress(self, atoms=None):
        return np.zeros(6)

    def calculation_required(self, atoms, quantities):
        return False

@pytest.mark.skipif(not shutil.which("pw.x"), reason="pw.x not found in PATH")
def test_real_loop_end_to_end(tmp_path):
    """
    Test the full loop with real pw.x if available.
    """
    # 1. Setup Config
    # Create a minimal config
    conf = OmegaConf.create({
        "experiment": {
            "name": "IntegrationTest",
            "max_cycles": 1,
            "box_size": 10.0,
            "work_dir": str(tmp_path)
        },
        "system": {
            "device": "cpu",
            "dft_command": "pw.x",
            "sssp_dir": "data/sssp", # Assuming this exists or we need to point to it
            "sssp_json": "SSSP_1.3.0_PBE_precision.json"
        },
        "training": {
            "energy_weight": 1.0,
            "forces_weight": 10.0
        },
        "exploration": {
            "md_steps": 5, # Very short for test
            "temperature": 100,
            "uncertainty_threshold": 0.0 # Force active learning (always uncertain)
        }
    })

    # Ensure SSSP exists (it should have been set up by script, but we are in a test)
    # If data/sssp doesn't exist, we might fail.
    # The prompt says "Task 1: SSSP Downloader" and "Task 5: Integration Test".
    # We should assume the user has run the downloader or we can try to run it/mock it.
    # But `QeOracle` checks for file existence.
    # If the environment doesn't have SSSP, we should probably skip or fail.
    # For now, let's assume if pw.x exists, the user might have set up the env?
    # Actually, we can just point to a fake SSSP if we want to test logic, but this is a "Real Loop" test.
    # Let's rely on the `check_env` assumption.

    sssp_path = Path("data/sssp")
    sssp_json = sssp_path / "SSSP_1.3.0_PBE_precision.json"

    if not sssp_json.exists():
        pytest.skip("SSSP data not found. Run scripts/download_sssp.py first.")

    # 2. Components
    potential = MockPotential()

    # We use Real Oracle
    oracle = QeOracle(
        dft_command=conf.system.dft_command,
        pseudo_dir=sssp_path,
        sssp_json_path=sssp_json,
        kpts_density=0.1
    )

    generator = MockGenerator()

    # 3. Orchestrator
    orchestrator = ActiveLearningOrchestrator(
        config=conf,
        potential=potential,
        oracle=oracle,
        generator=generator
    )

    # 4. Run
    orchestrator.run_loop()

    # 5. Assertions
    assert len(orchestrator.dataset) > 0, "Dataset should have grown"
    assert (tmp_path / "model_cycle_1.pt").exists(), "Model checkpoint should exist"
    # Check if we actually ran DFT
    # The dataset elements should have 'calculator' as SinglePointCalculator
    # and contain energy/forces
    last_atoms = orchestrator.dataset[-1]
    assert last_atoms.calc is not None
    # We can't easily check results values without knowing QE output, but we know it didn't crash.
