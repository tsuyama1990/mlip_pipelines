from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms

from src.domain_models.config import DistillationConfig
from src.oracles.base import BaseOracle
from src.oracles.mace_manager import MACEManager
from src.oracles.tiered_oracle import TieredOracle


class MockPrimaryOracle(BaseOracle):
    def compute_batch(self, structures: list[Atoms], calc_dir: Path) -> list[Atoms]:
        results = []
        for i, atoms in enumerate(structures):
            evaluated = atoms.copy()  # type: ignore[no-untyped-call]
            evaluated.info['energy'] = -1.0
            evaluated.arrays['forces'] = np.zeros((len(atoms), 3))

            # First 5 are confident, last 5 are uncertain
            if i < 5:
                evaluated.info['mace_uncertainty'] = 0.01
            else:
                evaluated.info['mace_uncertainty'] = 0.10

            results.append(evaluated)
        return results

class MockPrimaryOracleMissingUncertainty(BaseOracle):
    def compute_batch(self, structures: list[Atoms], calc_dir: Path) -> list[Atoms]:
        results = []
        for atoms in structures:
            evaluated = atoms.copy()  # type: ignore[no-untyped-call]
            evaluated.info['energy'] = -1.0
            evaluated.arrays['forces'] = np.zeros((len(atoms), 3))
            results.append(evaluated)
        return results

class MockFallbackOracle(BaseOracle):
    def compute_batch(self, structures: list[Atoms], calc_dir: Path) -> list[Atoms]:
        results = []
        for atoms in structures:
            evaluated = atoms.copy()  # type: ignore[no-untyped-call]
            evaluated.info['energy'] = -100.0  # distinctive energy
            evaluated.arrays['forces'] = np.ones((len(atoms), 3))
            evaluated.info['from_fallback'] = True
            results.append(evaluated)
        return results


def test_tiered_oracle_routing(tmp_path: Path) -> None:
    primary = MockPrimaryOracle()
    fallback = MockFallbackOracle()

    # Threshold is 0.05.
    # 5 structures have 0.01, so they bypass fallback.
    # 5 structures have 0.10, so they hit fallback.
    tiered = TieredOracle(primary, fallback, threshold=0.05)

    structures = [Atoms("Fe", positions=[(0, 0, 0)]) for _ in range(10)]

    results = tiered.compute_batch(structures, tmp_path)

    assert len(results) == 10

    # Check that exactly 5 hit the fallback
    fallback_count = sum(1 for atoms in results if atoms.info.get('from_fallback', False))
    assert fallback_count == 5

    # Check that exactly 5 bypassed fallback
    bypassed_count = sum(1 for atoms in results if not atoms.info.get('from_fallback', False))
    assert bypassed_count == 5


def test_tiered_oracle_missing_uncertainty(tmp_path: Path) -> None:
    primary = MockPrimaryOracleMissingUncertainty()
    fallback = MockFallbackOracle()

    tiered = TieredOracle(primary, fallback, threshold=0.05)
    structures = [Atoms("Fe", positions=[(0, 0, 0)]) for _ in range(3)]

    results = tiered.compute_batch(structures, tmp_path)

    assert len(results) == 3

    # Check that ALL hit the fallback because uncertainty was missing
    fallback_count = sum(1 for atoms in results if atoms.info.get('from_fallback', False))
    assert fallback_count == 3


def test_mace_manager_success(tmp_path: Path) -> None:
    config = DistillationConfig(mace_model_path="mace-mp-0-medium")

    class MockMACECalculator:
        def __init__(self, **kwargs: dict) -> None:
            pass

    import sys
    from unittest import mock

    mock_mace = mock.MagicMock()
    mock_mace.calculators.mace_mp.mace_mp.return_value = MockMACECalculator()
    sys.modules["mace"] = mock_mace
    sys.modules["mace.calculators"] = mock_mace.calculators
    sys.modules["mace.calculators.mace_mp"] = mock_mace.calculators.mace_mp

    try:
        manager = MACEManager(config)

        atoms = Atoms("Fe", positions=[(0, 0, 0)])

        # We need to mock get_potential_energy, get_forces and the calc.results dictionary
        def mock_get_potential_energy(self):
            self.calc.results = {"mace_uncertainty": 0.02}
            return -5.0

        def mock_get_forces(self):
            return np.zeros((1, 3))

        with patch("ase.Atoms.get_potential_energy", mock_get_potential_energy):
            with patch("ase.Atoms.get_forces", mock_get_forces):
                results = manager.compute_batch([atoms], tmp_path)

                assert len(results) == 1
                res = results[0]
                assert res.info["energy"] == -5.0
                assert np.array_equal(res.arrays["forces"], np.zeros((1, 3)))
                assert res.info["mace_uncertainty"] == 0.02
    finally:
        del sys.modules["mace"]
        del sys.modules["mace.calculators"]
        del sys.modules["mace.calculators.mace_mp"]


def test_mace_manager_missing_uncertainty(tmp_path: Path) -> None:
    config = DistillationConfig(mace_model_path="mace-mp-0-medium")

    class MockMACECalculator:
        def __init__(self, **kwargs: dict) -> None:
            pass

    import sys
    from unittest import mock

    mock_mace = mock.MagicMock()
    mock_mace.calculators.mace_mp.mace_mp.return_value = MockMACECalculator()
    sys.modules["mace"] = mock_mace
    sys.modules["mace.calculators"] = mock_mace.calculators
    sys.modules["mace.calculators.mace_mp"] = mock_mace.calculators.mace_mp

    try:
        manager = MACEManager(config)

        atoms = Atoms("Fe", positions=[(0, 0, 0)])

        def mock_get_potential_energy(self):
            self.calc.results = {}  # Missing uncertainty
            return -5.0

        def mock_get_forces(self):
            return np.zeros((1, 3))

        with patch("ase.Atoms.get_potential_energy", mock_get_potential_energy):
            with patch("ase.Atoms.get_forces", mock_get_forces):
                with pytest.raises(ValueError, match="MACEManager failed to extract uncertainty metric"):
                    manager.compute_batch([atoms], tmp_path)
    finally:
        del sys.modules["mace"]
        del sys.modules["mace.calculators"]
        del sys.modules["mace.calculators.mace_mp"]
