from unittest.mock import Mock, patch

import pytest
from ase import Atoms

from src.domain_models.config import DistillationConfig, ProjectConfig
from src.oracles.mace_manager import MACEManager


@pytest.fixture
def mock_config():
    config = Mock(spec=ProjectConfig)
    config.distillation_config = Mock(spec=DistillationConfig)
    config.distillation_config.mace_model_path = "mace-mp-0-medium"
    return config


def test_mace_manager_compute_batch(mock_config, tmp_path):
    manager = MACEManager(mock_config)

    from typing import Any

    class MockMACECalc:
        def __init__(self, **kwargs: Any) -> None:
            self.results: dict[str, Any] = {}

    mock_calc_instance = MockMACECalc()
    manager._calc = mock_calc_instance

    atoms1 = Atoms("Fe")
    atoms2 = Atoms("Fe")

    def mock_get_potential_energy():
        return -5.0

    def mock_get_forces():
        import numpy as np
        return np.array([[0.1, 0.2, 0.3]])

    with (
        patch("ase.Atoms.get_potential_energy", side_effect=mock_get_potential_energy),
        patch("ase.Atoms.get_forces", side_effect=mock_get_forces),
    ):
        # Case 1: valid uncertainty via 'mace_uncertainty'
        mock_calc_instance.results = {"mace_uncertainty": 0.02}
        results = manager.compute_batch([atoms1], tmp_path)
        assert len(results) == 1
        assert results[0].info["energy"] == -5.0
        assert "forces" in results[0].arrays
        assert results[0].info["mace_uncertainty"] == 0.02

        # Case 2: missing uncertainty
        mock_calc_instance.results = {"energy": -5.0}
        with pytest.raises(ValueError, match="MACE calculator failed to produce an uncertainty metric"):
            manager.compute_batch([atoms2], tmp_path)

        # Case 3: valid uncertainty via 'node_energy_variance'
        import numpy as np
        mock_calc_instance.results = {"node_energy_variance": np.array([0.01, 0.05])}
        results = manager.compute_batch([atoms1], tmp_path)
        assert len(results) == 1
        assert results[0].info["mace_uncertainty"] == 0.05
