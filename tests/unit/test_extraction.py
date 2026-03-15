import typing

import numpy as np
from ase import Atoms
from ase.build import bulk, molecule
from ase.calculators.calculator import Calculator, all_changes

from src.domain_models.config import CutoutConfig
from src.generators.extraction import (
    _extract_spherical_zones,
    _passivate_surface,
    _pre_relax_buffer,
    extract_intelligent_cluster,
)


class HookeanCalculatorMock(Calculator):
    implemented_properties: typing.ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        self.k = 1.0

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] | None = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)

        if atoms is None:
            return

        pos = atoms.positions
        com = atoms.get_center_of_mass()

        # Spring force pulling towards COM
        forces = -self.k * (pos - com)

        # Energy is sum of 0.5 * k * dx^2
        energy = np.sum(0.5 * self.k * np.sum((pos - com) ** 2, axis=1))

        self.results["forces"] = forces
        self.results["energy"] = energy


def test_extract_spherical_zones() -> None:
    # 10x10x10 simple cubic lattice
    atoms = bulk("Cu", "sc", a=1.0).repeat((10, 10, 10))

    # Target atom at index ~500 (middle of the box)
    center_idx = 555

    # Extract with core_r=1.1, buffer_r=2.1
    cluster = _extract_spherical_zones(atoms, center_idx, core_r=1.1, buffer_r=2.1)

    # Center atom + 6 nearest neighbors should be in core (7 atoms total)
    force_weights = cluster.arrays["force_weights"]
    assert np.sum(force_weights == 1.0) == 7

    # Check that there are more atoms in the buffer
    assert len(cluster) > 7
    assert np.sum(force_weights == 0.0) == len(cluster) - 7


def test_passivate_surface() -> None:
    # Simple diatomic molecule
    atoms = molecule("NaCl")

    # Manually assign force weights to mock extraction
    # core is Na (idx 0), buffer is Cl (idx 1)
    atoms.arrays["force_weights"] = np.array([1.0, 0.0])

    # Passivate
    initial_len = len(atoms)
    passivation_count = _passivate_surface(atoms, passivating_element="H")

    assert passivation_count == 0
    assert len(atoms) == initial_len


def test_passivate_surface_broken_bond() -> None:
    # Using MgO which is ionic and will definitely have missing bonds
    bulk_atoms = bulk("MgO", "rocksalt", a=4.21).repeat((3, 3, 3))

    # Middle Mg atom
    center_idx = 27

    # Small core and buffer to ensure surface atoms are undercoordinated
    cluster = _extract_spherical_zones(bulk_atoms, center_idx, core_r=2.6, buffer_r=4.5)

    passivation_count = _passivate_surface(cluster, passivating_element="H")

    assert passivation_count > 0

    fw = cluster.arrays["force_weights"]
    assert len(fw) == len(cluster)
    assert np.all(fw[-passivation_count:] == 0.0)


def test_pre_relax_buffer() -> None:
    atoms = Atoms(
        "Cu3",
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
    )
    # Atom 0 is core, 1 and 2 are buffer
    atoms.arrays["force_weights"] = np.array([1.0, 0.0, 0.0])

    initial_positions = atoms.positions.copy()

    mock_calc = HookeanCalculatorMock()

    _pre_relax_buffer(atoms, mock_calc)

    final_positions = atoms.positions

    # Core atom should not move
    np.testing.assert_allclose(final_positions[0], initial_positions[0])

    # Buffer atoms should move due to spring force pulling to center of mass
    assert not np.allclose(final_positions[1], initial_positions[1])
    assert not np.allclose(final_positions[2], initial_positions[2])


def test_extract_intelligent_cluster() -> None:
    bulk_atoms = bulk("Cu", "fcc", a=3.6).repeat((3, 3, 3))
    center_idx = 13

    config = CutoutConfig(
        core_radius=2.0,
        buffer_radius=3.0,
        enable_passivation=True,
        enable_pre_relaxation=True,
        passivation_element="H",
    )

    mock_calc = HookeanCalculatorMock()

    result = extract_intelligent_cluster(bulk_atoms, center_idx, config, mock_calc)

    assert result.cluster is not None
    assert "force_weights" in result.cluster.arrays
    assert result.passivation_atoms_added >= 0
