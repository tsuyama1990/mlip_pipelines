from ase import Atoms

from src.domain_models.config import OracleConfig
from src.oracles.dft_oracle import DFTManager


def test_dft_manager_initialization():
    config = OracleConfig()
    manager = DFTManager(config)
    assert manager.config.kspacing == 0.05


def test_periodic_embedding():
    config = OracleConfig()
    manager = DFTManager(config)
    atoms = Atoms("Fe", positions=[(0, 0, 0)])

    embedded = manager._apply_periodic_embedding(atoms)

    # Must be Orthorhombic cell for embedding
    cell = embedded.get_cell()
    assert cell[0][1] == 0.0
    assert cell[0][2] == 0.0
    assert cell[1][0] == 0.0
    assert cell[1][2] == 0.0
    assert cell[2][0] == 0.0
    assert cell[2][1] == 0.0


def test_compute_batch(monkeypatch, tmp_path):
    config = OracleConfig()
    manager = DFTManager(config)

    from typing import Any, ClassVar

    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator

    class MockCalc(Calculator):
        implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.parameters = {
                "input_data": {"electrons": {"mixing_beta": 0.7, "diagonalization": "david"}}
            }

        def calculate(self, atoms=None, properties=None, system_changes=None):
            if properties is None:
                properties = ["energy"]
            self.results = {
                "energy": -5.0,
                "forces": np.zeros((len(atoms) if atoms else 0, 3)),
                "stress": np.zeros(6),
            }

    atoms1 = Atoms("Fe", positions=[(0, 0, 0)])
    atoms2 = Atoms("Pt", positions=[(0, 0, 0)])

    # Monkeypatch the get_calculator logic to inject mock
    manager._get_calculator = lambda atoms, work_dir: MockCalc()

    results = manager.compute_batch([atoms1, atoms2], tmp_path)

    assert len(results) == 2
    assert "energy" in results[0].calc.results
    assert results[0].calc.results["energy"] == -5.0
