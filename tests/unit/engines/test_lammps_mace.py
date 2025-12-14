
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from ase import Atoms
import sys
import ctypes

# Mock the lammps module before importing the driver
sys.modules['lammps'] = MagicMock()

# Use absolute imports to match the application code
from src.engines.lammps_mace import LammpsMaceDriver
from src.core.exceptions import UncertaintyInterrupt

@pytest.fixture
def mock_potential():
    """Fixture for a mock MacePotential."""
    potential = MagicMock()
    # Mock predict to return (energy, forces, stress)
    # Default return: 2 atoms, 1.0 forces
    potential.predict.return_value = (0.0, np.ones((2, 3)), np.zeros((3, 3)))
    return potential

@pytest.fixture
def lammps_driver(mock_potential):
    """Fixture for a LammpsMaceDriver with a mock potential."""
    return LammpsMaceDriver(potential=mock_potential)

def test_callback_normal_step(lammps_driver, mock_potential):
    """Test the callback function during a normal MD step."""
    nlocal = 2
    # Create dummy atoms
    lammps_driver.current_atoms = Atoms('H2', positions=[[0,0,0], [1,1,1]])

    # Create numpy arrays and cast to ctypes pointers as expected by the driver
    x_np = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    x = x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    f_np = np.zeros((nlocal, 3), dtype=np.float64)
    f = f_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # Tags: 1, 2 (Identity)
    tags_np = np.array([1, 2], dtype=np.int32)
    tags = tags_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    mock_potential.get_uncertainty.return_value = np.array([0.0, 0.0])
    lammps_driver.threshold = 1.0

    lammps_driver._callback(None, 1, nlocal, tags, x, f)

    # Check that f_np was updated (since f points to its memory)
    assert np.array_equal(f_np, np.ones((2, 3)))
    mock_potential.predict.assert_called_once()

def test_callback_uncertainty_interrupt(lammps_driver, mock_potential):
    """Test that the callback raises UncertaintyInterrupt when uncertainty is high."""
    nlocal = 2
    lammps_driver.current_atoms = Atoms('H2', positions=[[0,0,0], [1,1,1]])

    x_np = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    x = x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    f_np = np.zeros((nlocal, 3), dtype=np.float64)
    f = f_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    tags_np = np.array([1, 2], dtype=np.int32)
    tags = tags_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    mock_potential.get_uncertainty.return_value = np.array([2.0, 0.5])
    lammps_driver.threshold = 1.0

    with pytest.raises(UncertaintyInterrupt) as excinfo:
        lammps_driver._callback(None, 1, nlocal, tags, x, f)

    assert isinstance(excinfo.value.atoms, Atoms)
    assert np.array_equal(excinfo.value.uncertainty, np.array([2.0, 0.5]))

def test_callback_sorting(lammps_driver, mock_potential):
    """Test that the callback handles scrambled atom order using tags."""
    nlocal = 2
    lammps_driver.current_atoms = Atoms('H2', positions=[[0,0,0], [1,1,1]])

    # Setup MACE return: Force on Atom 1 is [1,1,1], Force on Atom 2 is [2,2,2]
    # NOTE: MACE returns forces in ASE order (Atom 1, Atom 2).
    mock_potential.predict.return_value = (0.0, np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]), np.zeros((3, 3)))
    mock_potential.get_uncertainty.return_value = np.array([0.0, 0.0])

    # Scenario: LAMMPS has Atom 2 at index 0, and Atom 1 at index 1.
    # Tags: [2, 1]
    tags_np = np.array([2, 1], dtype=np.int32)
    tags = tags_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    # Coords passed from LAMMPS (scrambled)
    # Index 0 (Tag 2): [1,1,1]
    # Index 1 (Tag 1): [0,0,0]
    x_np = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    x = x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    f_np = np.zeros((nlocal, 3), dtype=np.float64)
    f = f_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    lammps_driver._callback(None, 1, nlocal, tags, x, f)

    # Check current_atoms positions - should be sorted back to ASE order
    # Atom 1 (from index 1): [0,0,0]
    # Atom 2 (from index 0): [1,1,1]
    expected_pos = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert np.allclose(lammps_driver.current_atoms.get_positions(), expected_pos)

    # Check forces in LAMMPS array - should be mapped back to scrambled order
    # Index 0 (Tag 2): Should have Force 2 -> [2,2,2]
    # Index 1 (Tag 1): Should have Force 1 -> [1,1,1]
    expected_f = np.array([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]])
    assert np.allclose(f_np, expected_f)

@patch('src.engines.lammps_mace.lammps')
def test_lammps_configuration(mock_lammps_constructor, mock_potential):
    """Test that the LAMMPS instance is configured correctly."""
    mock_lmp = MagicMock()
    mock_lammps_constructor.return_value = mock_lmp

    driver = LammpsMaceDriver(potential=mock_potential)
    atoms = Atoms('H2', positions=[[0,0,0], [0,0,1]], cell=[10,10,10])
    script = "fix 1 all nvt temp 300 300 0.1"

    with patch.object(driver, '_callback', autospec=True) as mock_callback:
        try:
            driver.run_md(atoms, script, 1.0)
        except Exception:
            pass

        mock_lmp.command.assert_any_call("pair_style none")
        mock_lmp.command.assert_any_call("fix MACE_FORCE all external pf/callback 1 1")
        mock_lmp.set_fix_external_callback.assert_called_once_with("MACE_FORCE", mock_callback, driver)
