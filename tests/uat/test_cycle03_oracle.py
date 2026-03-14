import pytest
from ase import Atoms
from src.oracles.dft_oracle import DFTManager
from src.domain_models.config import OracleConfig
from unittest.mock import patch
from pathlib import Path
from src.core.exceptions import OracleConvergenceError

def test_uat_03_01_successful_dft_calculation_and_embedding(tmp_path: Path):
    """
    UAT-03-01: Successful DFT Calculation and Embedding
    """
    config = OracleConfig(buffer_size=4.0)
    manager = DFTManager(config)

    # GIVEN a list of localized Atoms objects representing highly uncertain clusters
    # AND the DFTManager is configured with a buffer_size of 4.0 A
    cluster = Atoms("Fe", positions=[(0, 0, 0)])

    # Mocking the ASE calculator
    class MockCalc:
        def __init__(self) -> None:
            self.parameters = {"input_data": {"electrons": {}}}
            self.results = {"energy": -1.0, "forces": [[0,0,0]], "stress": [0,0,0,0,0,0]}
            self.name = "mock"

        def get_potential_energy(self, atoms=None):
            return -1.0

        def get_forces(self, atoms=None):
            return [[0,0,0]]

        def get_stress(self, atoms=None):
            return [0,0,0,0,0,0]

    with patch.object(manager, "_get_calculator", return_value=MockCalc()):
        with patch("ase.Atoms.get_potential_energy", return_value=-1.0), \
             patch("ase.Atoms.get_forces", return_value=[[0,0,0]]), \
             patch("ase.Atoms.get_stress", return_value=[0,0,0,0,0,0]):

            # WHEN compute_batch() is called with these structures
            results = manager.compute_batch([cluster], tmp_path)

            # THEN each structure should be processed by _apply_periodic_embedding()
            # AND the resulting Atoms objects passed to the calculator should have pbc=True
            assert len(results) == 1
            embedded = results[0]
            assert embedded.get_pbc().all() == True

            # AND the new cell dimensions should accurately encompass the original cluster plus the 4.0 A buffer on all sides
            cell = embedded.get_cell()
            assert cell[0][0] == 8.0  # 0 + 2 * 4.0
            assert cell[1][1] == 8.0
            assert cell[2][2] == 8.0

            # AND the atoms should be centered within this new periodic cell
            # Original min and max are 0. Center is 0. Lengths are 8. box_center is 4.
            # shifted_pos = pos + box_center - center = 0 + 4 - 0 = 4
            pos = embedded.get_positions()
            assert pos[0][0] == 4.0
            assert pos[0][1] == 4.0
            assert pos[0][2] == 4.0

def test_uat_03_02_self_healing_on_scf_convergence_failure(tmp_path: Path):
    """
    UAT-03-02: Self-Healing on SCF Convergence Failure
    """
    config = OracleConfig(max_retries=3)
    manager = DFTManager(config)

    # GIVEN a candidate structure that is known to cause SCF convergence issues (mocked for testing)
    # AND the DFTManager is configured with max_retries=3
    cluster = Atoms("Fe", positions=[(0, 0, 0)])

    class MockCalc:
        def __init__(self) -> None:
            self.parameters = {"input_data": {"electrons": {"mixing_beta": 0.7, "diagonalization": "david"}}}
            self.call_count = 0

    calc = MockCalc()

    with patch.object(manager, "_get_calculator", return_value=calc):
        def mock_get_potential_energy(self):
            calc.call_count += 1
            if calc.call_count == 1:
                # WHEN compute_batch() executes the first calculation attempt
                # THEN the calculator (mocked) should raise an ase.calculators.calculator.CalculationFailed error
                raise Exception("SCF Failed")
            return 1.0

        def mock_get_forces(self):
            return 1.0

        def mock_get_stress(self):
            return 1.0

        with patch("ase.Atoms.get_potential_energy", mock_get_potential_energy), patch(
            "ase.Atoms.get_forces", mock_get_forces
        ), patch("ase.Atoms.get_stress", mock_get_stress):

            results = manager.compute_batch([cluster], tmp_path)

            # AND the DFTManager should catch this exception
            # AND the mixing_beta parameter on the calculator should be automatically reduced (e.g., from 0.7 to 0.3)
            # AND the DFTManager should automatically retry the calculation
            assert len(results) == 1
            assert calc.parameters["input_data"]["electrons"]["mixing_beta"] == 0.3

            # AND the second attempt (mocked to succeed) should successfully return the structure with calculated forces and energies.
            assert calc.call_count == 2

def test_uat_03_03_exceeding_physical_constraints():
    """
    UAT-03-03: Exceeding Physical Constraints
    """
    config = OracleConfig(max_cell_dimension=100.0)
    manager = DFTManager(config)

    # GIVEN a massive candidate cluster where the maximum interatomic distance exceeds the max_cell_dimension
    # Example: interatomic distance is 150 > 100
    cluster = Atoms("Fe2", positions=[(0, 0, 0), (150, 0, 0)])

    # WHEN _apply_periodic_embedding() attempts to create a supercell for this structure
    # THEN the validation logic should immediately detect the violation
    # AND a ValueError should be raised, indicating the cell dimension is too large
    # AND the structure should be safely skipped or the batch calculation should halt, preventing OOM crashes.
    with pytest.raises(ValueError, match="exceed maximum allowed coordinates|too large"):
        # Depends on max_coord. If it fails max_coord first, it's also a ValueError.
        # Let's override max_coord to let it reach max_cell_dimension logic.
        manager.config.max_coord = 200.0
        manager._apply_periodic_embedding(cluster)
