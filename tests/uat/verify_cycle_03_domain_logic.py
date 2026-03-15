# ruff: noqa: N803, S108, T201
import sys
from pathlib import Path

# Inject current working directory into sys.path to resolve src imports
sys.path.insert(0, str(Path.cwd()))

import marimo

__generated_with = "0.10.19"
app = marimo.App()

@app.cell
def _():
    import numpy as np
    from ase import Atoms

    from src.domain_models.config import DistillationConfig
    from src.oracles.base import BaseOracle
    from src.oracles.mace_manager import MACEManager
    from src.oracles.tiered_oracle import TieredOracle

    class MockMACEManager(BaseOracle):
        def compute_batch(self, structures: list[Atoms], calc_dir: Path) -> list[Atoms]:
            results = []
            for i, atoms in enumerate(structures):
                evaluated = atoms.copy()  # type: ignore[no-untyped-call]
                evaluated.info['energy'] = -1.0
                evaluated.arrays['forces'] = np.zeros((len(atoms), 3))
                # 0-4 are confident, 5-9 are uncertain, 10-11 lack uncertainty entirely
                if i < 5:
                    evaluated.info['mace_uncertainty'] = 0.01
                elif i < 10:
                    evaluated.info['mace_uncertainty'] = 0.10
                results.append(evaluated)
            return results

    class MockDFTManager(BaseOracle):
        def compute_batch(self, structures: list[Atoms], calc_dir: Path) -> list[Atoms]:
            results = []
            for atoms in structures:
                evaluated = atoms.copy()  # type: ignore[no-untyped-call]
                evaluated.info['energy'] = -100.0
                evaluated.arrays['forces'] = np.ones((len(atoms), 3))
                evaluated.info['dft_calculated'] = True
                results.append(evaluated)
            return results

    return (
        Path,
        Atoms,
        DistillationConfig,
        BaseOracle,
        MACEManager,
        TieredOracle,
        MockMACEManager,
        MockDFTManager,
        np,
    )

@app.cell
def test_uat_c03_01(TieredOracle, MockMACEManager, MockDFTManager, Atoms, Path) -> tuple:
    # UAT-C03-01: Zero-Shot Distillation Structure Acceptance and Fast-Tracking
    _primary = MockMACEManager()
    _fallback = MockDFTManager()
    _tiered = TieredOracle(_primary, _fallback, threshold=0.05)

    _structures = [Atoms("Fe", positions=[(0, 0, 0)]) for _ in range(5)]
    _results = _tiered.compute_batch(_structures, Path("/tmp"))

    # 5 structures passed in, all have uncertainty 0.01 (which < 0.05).
    # DFT shouldn't be called.
    assert len(_results) == 5
    for _r in _results:
        assert not _r.info.get("dft_calculated", False)

    print("UAT-C03-01 Passed: Confident structures bypass DFT correctly.")
    return ()

@app.cell
def test_uat_c03_02(TieredOracle, MockMACEManager, MockDFTManager, Atoms, Path) -> tuple:
    # UAT-C03-02: Tiered Oracle Routing of High-Uncertainty Structures to DFT
    _primary = MockMACEManager()
    _fallback = MockDFTManager()
    _tiered = TieredOracle(_primary, _fallback, threshold=0.05)

    _structures = [Atoms("Fe", positions=[(0, 0, 0)]) for _ in range(10)]
    _results = _tiered.compute_batch(_structures, Path("/tmp"))

    # 10 structures. 5 have 0.01, 5 have 0.10.
    # 5 should bypass, 5 should be calculated by DFT.
    _dft_count = sum(1 for r in _results if r.info.get("dft_calculated", False))

    assert len(_results) == 10
    assert _dft_count == 5

    print("UAT-C03-02 Passed: High-uncertainty structures are correctly routed to DFT.")
    return ()

@app.cell
def test_uat_c03_03(MACEManager, DistillationConfig, Atoms, Path, np) -> tuple:
    # UAT-C03-03: Validation of MACE Epistemic Uncertainty Extraction and Normalization
    from unittest import mock

    _config = DistillationConfig(mace_model_path="mace-mp-0-medium")

    class MockCalculator:
        def __init__(self, **kwargs: dict) -> None:
            pass

    # Mocking at the sys.modules level because we want to intercept the internal import
    import sys
    mock_mace = mock.MagicMock()
    mock_mace.calculators.mace_mp.mace_mp.return_value = MockCalculator()
    sys.modules["mace"] = mock_mace
    sys.modules["mace.calculators"] = mock_mace.calculators
    sys.modules["mace.calculators.mace_mp"] = mock_mace.calculators.mace_mp

    try:
        _manager = MACEManager(_config)
        _atoms = Atoms("Fe", positions=[(0, 0, 0)])

        def mock_energy(self) -> float:
            self.calc.results = {"mace_uncertainty": 0.042}
            return -2.5

        def mock_forces(self) -> np.ndarray:
            return np.array([[0.1, 0.2, 0.3]])

        with mock.patch("ase.Atoms.get_potential_energy", mock_energy):
            with mock.patch("ase.Atoms.get_forces", mock_forces):
                _results = _manager.compute_batch([_atoms], Path("/tmp"))

                assert len(_results) == 1
                assert _results[0].info["mace_uncertainty"] == 0.042
                assert _results[0].info["energy"] == -2.5
                assert np.allclose(_results[0].arrays["forces"], [[0.1, 0.2, 0.3]])
    finally:
        del sys.modules["mace"]
        del sys.modules["mace.calculators"]
        del sys.modules["mace.calculators.mace_mp"]

    print("UAT-C03-03 Passed: MACE uncertainty, energy, and forces are successfully extracted.")
    return ()

@app.cell
def test_uat_c03_04(TieredOracle, MockMACEManager, MockDFTManager, Atoms, Path) -> tuple:
    # UAT-C03-04: Safety Fallback for Missing Uncertainty Metrics
    _primary = MockMACEManager()
    _fallback = MockDFTManager()
    _tiered = TieredOracle(_primary, _fallback, threshold=0.05)

    # Passing 12 structures: 0-4 (confident), 5-9 (uncertain), 10-11 (missing uncertainty)
    _structures = [Atoms("Fe", positions=[(0, 0, 0)]) for _ in range(12)]
    _results = _tiered.compute_batch(_structures, Path("/tmp"))

    # 5 confident. 5 uncertain + 2 missing = 7 sent to DFT.
    _dft_count = sum(1 for r in _results if r.info.get("dft_calculated", False))

    assert len(_results) == 12
    assert _dft_count == 7

    print("UAT-C03-04 Passed: Missing uncertainty metrics forcefully route to DFT fallback.")
    return ()

@app.cell
def test_uat_c03_05(TieredOracle, MockMACEManager, MockDFTManager, Atoms, Path) -> tuple:
    # UAT-C03-05: Validation of Fallback Queue Processing Metadata Preservation
    _primary = MockMACEManager()
    _fallback = MockDFTManager()
    _tiered = TieredOracle(_primary, _fallback, threshold=0.05)

    _structures = []
    # Create an atom at index 6 that will fail the threshold check
    for i in range(10):
        # We add some unique metadata to check preservation
        atom = Atoms("Pt" if i % 2 == 0 else "Pd", positions=[(i, i, i)])
        atom.info['original_id'] = i
        _structures.append(atom)

    _results = _tiered.compute_batch(_structures, Path("/tmp"))

    # 5 should have gone to DFT.
    for _r in _results:
        # Check metadata was preserved
        assert 'original_id' in _r.info
        assert _r.get_chemical_symbols()[0] in ["Pt", "Pd"]  # type: ignore[no-untyped-call]

    print("UAT-C03-05 Passed: Fallback queue correctly maintains atomic properties and metadata.")
    return ()

if __name__ == "__main__":
    app.run()
