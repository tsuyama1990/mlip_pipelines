import pytest
import numpy as np
from ase import Atoms
from core.engines.relaxer import StructureRelaxer

def test_relaxer_run_mocked(mocker, default_settings, test_atoms):
    """
    Test StructureRelaxer.run with a mocked calculator.
    Verifies logic without heavy computation.
    """
    # 1. Mock the Calculator
    # We mock the instance that will be attached to atoms
    mock_calc = mocker.Mock()

    # Setup return values for get_potential_energy and get_forces
    # Call 1: Initial state
    # Call 2...N: Optimization steps
    mock_calc.get_potential_energy.side_effect = [-10.0, -10.5, -11.0]

    # Forces: Need to match atoms length
    n_atoms = len(test_atoms)

    # Use side_effect function to return forces indefinitely
    def get_forces_side_effect(*args, **kwargs):
        # Determine behavior based on call count if needed, or just return converging forces
        # We can just return a small force to simulate convergence after a few steps
        # But LBFGS might check multiple times.
        # Let's return decreasing forces based on call count stored in mock
        count = mock_calc.get_forces.call_count
        if count < 2:
            return np.ones((n_atoms, 3)) * 0.5
        elif count < 5:
            return np.ones((n_atoms, 3)) * 0.1
        else:
            return np.zeros((n_atoms, 3))

    mock_calc.get_forces.side_effect = get_forces_side_effect

    # Also fix energy to not run out
    mock_calc.get_potential_energy.side_effect = None
    mock_calc.get_potential_energy.return_value = -10.0

    # Attach mock calc
    test_atoms.calc = mock_calc

    # 2. Initialize Relaxer
    # Use small steps to ensure it doesn't run forever if mock fails
    default_settings.relax.steps = 5
    relaxer = StructureRelaxer(default_settings)

    # 3. Run
    result = relaxer.run(test_atoms, run_id="test_mock_run")

    # 4. Verify
    assert result["run_id"] == "test_mock_run"
    assert "final_structure" in result
    assert "trajectory" in result
    assert len(result["trajectory"]) > 0

    # Check if optimize was called (implicitly via calc calls)
    assert mock_calc.get_potential_energy.called
    assert mock_calc.get_forces.called
