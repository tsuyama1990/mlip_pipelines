
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from ase import Atoms
import sys

# Mock the lammps module before importing the driver
sys.modules['lammps'] = MagicMock()

from engines.lammps_mace import LammpsMaceDriver
from core.exceptions import UncertaintyInterrupt

@pytest.fixture
def mock_potential():
    """Fixture for a mock MacePotential."""
    potential = MagicMock()
    potential.predict.return_value = (0.0, np.ones((2, 3)), np.zeros((3, 3)))
    return potential

@pytest.fixture
def lammps_driver(mock_potential):
    """Fixture for a LammpsMaceDriver with a mock potential."""
    return LammpsMaceDriver(potential=mock_potential)

def test_callback_normal_step(lammps_driver, mock_potential):
    """Test the callback function during a normal MD step."""
    nlocal = 2
    x = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).ctypes.data
    f = np.zeros((nlocal, 3)).ctypes.data

    mock_potential.get_uncertainty.return_value = np.array([0.0, 0.0])
    lammps_driver.threshold = 1.0

    lammps_driver._callback(None, 1, nlocal, None, x, f)

    forces_array = np.ctypeslib.as_array(f, shape=(nlocal, 3))
    assert np.array_equal(forces_array, np.ones((2, 3)))
    mock_potential.predict.assert_called_once()

def test_callback_uncertainty_interrupt(lammps_driver, mock_potential):
    """Test that the callback raises UncertaintyInterrupt when uncertainty is high."""
    nlocal = 2
    x = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).ctypes.data
    f = np.zeros((nlocal, 3)).ctypes.data

    mock_potential.get_uncertainty.return_value = np.array([2.0, 0.5])
    lammps_driver.threshold = 1.0

    with pytest.raises(UncertaintyInterrupt) as excinfo:
        lammps_driver._callback(None, 1, nlocal, None, x, f)

    assert isinstance(excinfo.value.atoms, Atoms)
    assert np.array_equal(excinfo.value.uncertainty, np.array([2.0, 0.5]))

@patch('engines.lammps_mace.lammps')
def test_lammps_configuration(mock_lammps_constructor, mock_potential):
    """Test that the LAMMPS instance is configured correctly."""
    mock_lmp = MagicMock()
    mock_lammps_constructor.return_value = mock_lmp

    driver = LammpsMaceDriver(potential=mock_potential)
    atoms = Atoms('H2', positions=[[0,0,0], [0,0,1]], cell=[10,10,10])
    script = "fix 1 all nvt temp 300 300 0.1"

    with patch.object(driver, '_callback'):
        try:
            driver.run_md(atoms, script, 1.0)
        except Exception:
            # We expect an error because the mock lmp object doesn't behave like the real one
            # but we can still check the calls.
            pass

    mock_lmp.command.assert_any_call("pair_style none")
    mock_lmp.command.assert_any_call("fix MACE_FORCE all external pf/callback 1 1")
    mock_lmp.set_fix_external_callback.assert_called_once_with("MACE_FORCE", driver._callback, driver)
    mock_lmp.command.assert_any_call(script)
