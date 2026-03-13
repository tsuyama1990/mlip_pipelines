from typing import Any

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator

from src.validators.stability_tests import check_mechanical_stability, check_phonopy_stability


class MockStableCalculator(Calculator):  # type: ignore[misc]
    implemented_properties: list[str] = ["energy", "forces", "stress"]  # noqa: RUF012

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ) -> None:  # type: ignore[override]
        super().calculate(atoms, properties, system_changes)

        # Simulate E = E0 + 1/2 V0 B s^2 logic perfectly
        # For a truly stable structure, we just need the energies to form a positive parabola
        # based on the strains applied in the tests.

        # Check what kind of strain is applied by looking at cell differences
        if atoms is None:
            self.results = {"energy": 0.0, "forces": np.zeros((1, 3)), "stress": np.zeros(6)}
            return

        cell = atoms.get_cell()
        vol = atoms.get_volume()

        # Simple heuristic to make E > E0 always if cell != standard
        E0 = -100.0
        # If it's a 3.5x3.5x3.5 perfect cube
        a = cell[0][0]
        cell[1][1]
        cell[2][2]

        # Just return a simple harmonic potential energy based on deviation from a perfect cube
        # to ensure positive coefficients
        s_vol = (vol - 3.5**3) / (3.5**3)
        s_tet = a / 3.5 - 1.0
        s_shear = cell[0][1] / 3.5

        energy = E0 + 100 * s_vol**2 + 50 * s_tet**2 + 30 * s_shear**2

        self.results = {
            "energy": energy,
            "forces": np.zeros((len(atoms), 3)),
            "stress": np.zeros(6),
        }


class MockUnstableCalculator(MockStableCalculator):
    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ) -> None:  # type: ignore[override]
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            return

        cell = atoms.get_cell()
        vol = atoms.get_volume()

        s_vol = (vol - 3.5**3) / (3.5**3)
        s_tet = cell[0][0] / 3.5 - 1.0

        # Negative parabola for shear makes C44 < 0
        # By ensuring a strong negative coefficient on the exact shear term
        # applied (cell[0][1] = 1.75 * s), we force the C44 polynomial fit to yield < 0.
        E0 = -100.0
        energy = E0 + 100 * s_vol**2 + 50 * s_tet**2 - 1000 * (cell[0][1] / 3.5) ** 2

        self.results["energy"] = energy


def test_mechanical_stability_stable() -> None:
    from ase.build import bulk

    atoms = bulk("Fe", "bcc", a=3.5)
    calc = MockStableCalculator()

    is_stable = check_mechanical_stability(atoms, calc)
    assert is_stable is True


def test_mechanical_stability_unstable() -> None:
    from ase.build import bulk

    atoms = bulk("Fe", "bcc", a=3.5)
    calc = MockUnstableCalculator()

    is_stable = check_mechanical_stability(atoms, calc)
    assert is_stable is False


def test_phonopy_stability_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    # We mock phonopy completely to avoid heavy calculations in unit tests
    class MockPhonopy:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.supercells_with_displacements = [None]

        def generate_displacements(self, distance: float) -> None:
            return None

        def produce_force_constants(self, forces: Any) -> None:
            return None

        def run_mesh(self, mesh: Any) -> None:
            return None

        def get_mesh_dict(self) -> dict[str, Any]:
            # Return positive frequencies for stable
            return {"frequencies": np.array([0.1, 0.2, 0.3])}

    # We need to mock the entire phonopy module if it's imported inside the function
    import phonopy

    monkeypatch.setattr(phonopy, "Phonopy", MockPhonopy)

    from ase.build import bulk

    atoms = bulk("Fe", "bcc", a=3.5)
    calc = MockStableCalculator()

    # We also mock PhonopyAtoms to avoid phonopy dependencies inside the test
    class MockPhonopyAtoms:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            return None

    monkeypatch.setattr(phonopy.structure.atoms, "PhonopyAtoms", MockPhonopyAtoms)

    is_stable = check_phonopy_stability(atoms, calc)
    assert is_stable is True


def test_phonopy_stability_unstable_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    class MockPhonopyUnstable:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.supercells_with_displacements = [None]

        def generate_displacements(self, distance: float) -> None:
            return None

        def produce_force_constants(self, forces: Any) -> None:
            return None

        def run_mesh(self, mesh: Any) -> None:
            return None

        def get_mesh_dict(self) -> dict[str, Any]:
            # Return negative (imaginary) frequencies
            return {"frequencies": np.array([-0.1, 0.2, 0.3])}

    import phonopy

    monkeypatch.setattr(phonopy, "Phonopy", MockPhonopyUnstable)

    class MockPhonopyAtomsUnstable:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            return None

    monkeypatch.setattr(phonopy.structure.atoms, "PhonopyAtoms", MockPhonopyAtomsUnstable)

    from ase.build import bulk

    atoms = bulk("Fe", "bcc", a=3.5)
    calc = MockStableCalculator()

    is_stable = check_phonopy_stability(atoms, calc)
    assert is_stable is False
