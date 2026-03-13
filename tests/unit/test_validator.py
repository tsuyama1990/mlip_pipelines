from pathlib import Path

import pytest

from src.domain_models.config import ValidatorConfig
from src.validators.validator import Validator


def test_validator_initialization() -> None:
    config = ValidatorConfig(energy_rmse_threshold=0.01)
    validator = Validator(config)
    assert validator.config.energy_rmse_threshold == 0.01


def test_validate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = ValidatorConfig()
    validator = Validator(config)

    # Needs a mock potential path
    dummy_pot = tmp_path / "dummy.yace"
    dummy_pot.write_text("elements version")

    from typing import Any, ClassVar

    from ase import Atoms
    from ase.calculators.calculator import Calculator

    class MockCalc(Calculator):  # type: ignore[misc]
        implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]

        def calculate(
            self,
            atoms: Atoms | None = None,
            properties: list[str] | None = None,
            system_changes: Any = None,
        ) -> None:
            import numpy as np

            if properties is None:
                properties = ["energy"]
            self.results = {
                "energy": -5.0,
                "forces": np.zeros((len(atoms) if atoms else 0, 3)),
                "stress": np.zeros(6),
            }

    import sys

    class MockPyacemaker:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __call__(self, *args: Any, **kwargs: Any) -> MockCalc:
            return MockCalc()

    # Mock phonopy
    class MockPhonopy:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.supercells_with_displacements: list[Any] = []

        def generate_displacements(self, distance: float) -> None:
            pass

        def produce_force_constants(self, forces: Any = None) -> None:
            pass

        def run_mesh(self, mesh: Any) -> None:
            pass

        def get_mesh_dict(self) -> dict[str, Any]:
            import numpy as np

            return {"frequencies": np.array([1.0, 2.0])}

    class MockPhonopyAtoms:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    from types import ModuleType

    # Create dummy modules for pyacemaker and phonopy
    pyacemaker_calc_mod = ModuleType("pyacemaker.calculator")
    setattr(pyacemaker_calc_mod, "pyacemaker", MockPyacemaker())  # type: ignore[attr-defined]
    sys.modules["pyacemaker.calculator"] = pyacemaker_calc_mod

    phonopy_mod = ModuleType("phonopy")
    phonopy_atoms_mod = ModuleType("phonopy.structure.atoms")
    phonopy_mod.Phonopy = MockPhonopy  # type: ignore[attr-defined]
    phonopy_atoms_mod.PhonopyAtoms = MockPhonopyAtoms  # type: ignore[attr-defined]
    sys.modules["phonopy"] = phonopy_mod
    sys.modules["phonopy.structure.atoms"] = phonopy_atoms_mod

    report = validator.validate(dummy_pot)

    assert hasattr(report, "energy_rmse")
    # Since our mock calc always returns -5.0, E_hydro will be equal to true_energy, not less, so it should be mechanically stable
    assert report.mechanically_stable is True
