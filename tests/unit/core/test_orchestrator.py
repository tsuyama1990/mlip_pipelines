
import pytest
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf
from ase import Atoms

from core.orchestrator import ActiveLearningOrchestrator
from core.exceptions import UncertaintyInterrupt

@pytest.fixture
def mock_config():
    """Fixture for a mock OmegaConf config."""
    return OmegaConf.create({
        "experiment": {
            "max_cycles": 2,
            "n_initial_seeds": 1,
            "exploration": {
                "lammps_script": "some_script",
                "threshold_ratio": 1.5,
            },
        },
    })

@pytest.fixture
def mock_components(mock_config):
    """Fixture to create mock components for the orchestrator."""
    potential = MagicMock()
    oracle = MagicMock()
    carver = MagicMock()
    generator = MagicMock()

    # Setup generator to return a valid Atoms object
    generator.generate_initial_pool.return_value = [Atoms('H')]

    return mock_config, potential, oracle, carver, generator

@patch('core.orchestrator.LammpsMaceDriver')
def test_loop_flow(mock_driver_constructor, mock_components):
    """Test the main control flow of the orchestrator."""
    mock_config, potential, oracle, carver, generator = mock_components

    # Setup mock driver behavior
    mock_driver = MagicMock()
    mock_driver_constructor.return_value = mock_driver
    mock_driver.run_md.side_effect = [
        (Atoms('H'), "FINISHED"),
        (Atoms('H2'), "UNCERTAIN"),
    ]

    orchestrator = ActiveLearningOrchestrator(
        config=mock_config,
        potential=potential,
        oracle=oracle,
        carver=carver,
        generator=generator,
    )

    orchestrator.run_loop()

    assert mock_driver.run_md.call_count == 2
    carver.carve.assert_called_once()
    oracle.compute.assert_called_once()
    potential.train.assert_called() # Called for initial seeds + uncertain candidate
    assert orchestrator.cycle_count == 2
