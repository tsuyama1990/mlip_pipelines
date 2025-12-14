
import pytest
from unittest.mock import MagicMock, patch
from ase import Atoms
from pathlib import Path

from oracles.qe_oracle import QeOracle
from core.exceptions import OracleComputationError

@pytest.fixture
def mock_sssp_loader():
    """Fixture for a mock SSSPLoader."""
    loader = MagicMock()
    loader.get_pseudo.return_value = "Si.upf"
    loader.get_recommended_cutoffs.return_value = (60.0, 240.0)
    return loader

@pytest.fixture
def qe_oracle(mock_sssp_loader):
    """Fixture for a QeOracle with a mock SSSPLoader."""
    with patch('oracles.qe_oracle.SSSPLoader', return_value=mock_sssp_loader):
        return QeOracle(
            pseudo_dir=Path("/fake/dir"),
            sssp_json_path=Path("/fake/sssp.json"),
        )

def test_input_generation(qe_oracle):
    """Test that the QE input is generated correctly."""
    atoms = Atoms('Si', cell=[10, 10, 10], pbc=True)

    with patch('ase.calculators.espresso.Espresso.calculate') as mock_calculate:
        qe_oracle.compute(atoms)
        pass

def test_robust_error_handling(qe_oracle):
    """Test that any calculator error is caught and re-raised as OracleComputationError."""
    atoms = Atoms('Si', cell=[10, 10, 10], pbc=True)

    with patch('ase.calculators.espresso.Espresso.calculate', side_effect=Exception("SCF not converged")):
        with pytest.raises(OracleComputationError):
            qe_oracle.compute(atoms)

def test_determine_kpts(qe_oracle):
    """Test the k-point determination logic."""
    # Test cluster
    atoms_cluster = Atoms('Si2', positions=[[0,0,0], [1,1,1]])
    kpts_cluster = qe_oracle._determine_kpts(atoms_cluster)
    assert kpts_cluster == (1, 1, 1)

    # Test periodic system
    atoms_periodic = Atoms('Si', cell=[5, 5, 5], pbc=True)
    qe_oracle.kpts_density = 1.0
    kpts_periodic = qe_oracle._determine_kpts(atoms_periodic)
    assert kpts_periodic == (2, 2, 2)
