
import pytest
from unittest.mock import MagicMock, patch
from ase import Atoms
from pathlib import Path
import numpy as np

from oracles.qe_oracle import QeOracle
from core.exceptions import OracleComputationError

MOCK_SSSP_DB = {
    'Si': {
        'filename': 'Si.upf',
        'cutoff_wfc': 30.0,
        'cutoff_rho': 120.0
    }
}

@pytest.fixture
def qe_oracle():
    """Fixture for a QeOracle with mocked SSSP loading and validation."""

    # Mock the sssp_json_path to appear as existing
    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = True
    mock_path.__str__.return_value = "/fake/sssp.json"
    mock_path.resolve.return_value = Path("/fake/sssp.json")

    # We patch load_sssp_db to return our mock DB
    # We patch validate_pseudopotentials to do nothing (pass validation)
    # We patch EspressoProfile because the environment's ASE version has a different signature
    with patch('oracles.qe_oracle.load_sssp_db', return_value=MOCK_SSSP_DB), \
         patch('oracles.qe_oracle.validate_pseudopotentials'), \
         patch('oracles.qe_oracle.EspressoProfile') as MockEspressoProfile:

        # Configure MockEspressoProfile to return a mock object
        # This prevents the TypeError during instantiation
        MockEspressoProfile.return_value = MagicMock()

        oracle = QeOracle(
            pseudo_dir=Path("/fake/dir"),
            sssp_json_path=mock_path,
        )
        yield oracle

def test_input_generation(qe_oracle):
    """Test that the QE input is generated correctly."""
    atoms = Atoms('Si', cell=[10, 10, 10], pbc=True)

    # We patch the methods of the Espresso class to avoid real execution
    with patch('ase.calculators.espresso.Espresso.get_potential_energy', return_value=-100.0) as mock_pe, \
         patch('ase.calculators.espresso.Espresso.get_forces', return_value=np.zeros((1, 3))) as mock_forces, \
         patch('ase.calculators.espresso.Espresso.get_stress', return_value=np.zeros(6)) as mock_stress:

        result = qe_oracle.compute(atoms)

        # Verify result is an Atoms object with SinglePointCalculator
        assert isinstance(result, Atoms)
        assert result.calc is not None
        assert result.get_potential_energy() == -100.0

        # Verify Espresso was called (implicitly by checking if mocked methods were called)
        mock_pe.assert_called_once()

def test_robust_error_handling(qe_oracle):
    """Test that any calculator error is caught and re-raised as OracleComputationError."""
    atoms = Atoms('Si', cell=[10, 10, 10], pbc=True)

    with patch('ase.calculators.espresso.Espresso.get_potential_energy', side_effect=Exception("SCF not converged")):
        with pytest.raises(OracleComputationError):
            qe_oracle.compute(atoms)

def test_determine_kpts(qe_oracle):
    """Test the k-point determination logic."""
    # Test cluster
    atoms_cluster = Atoms('Si2', positions=[[0,0,0], [1,1,1]])
    kpts_cluster = qe_oracle._determine_kpts(atoms_cluster)
    assert kpts_cluster == (1, 1, 1)

    # Test periodic system
    # Cell 5x5x5 A. Reciprocal vec len b = 1/5 = 0.2 A^-1 (using crystallographer definition in ASE)
    atoms_periodic = Atoms('Si', cell=[5, 5, 5], pbc=True)

    # Set density such that ceil(density * 0.2) = 2
    # 6.0 * 0.2 = 1.2 -> ceil(1.2) = 2
    qe_oracle.kpts_density = 6.0
    kpts_periodic = qe_oracle._determine_kpts(atoms_periodic)
    assert kpts_periodic == (2, 2, 2)
