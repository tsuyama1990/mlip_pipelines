from ase.build import bulk

from src.domain_models.config import StructureGeneratorConfig
from src.generators.defect_builder import DefectBuilder


def test_apply_vacancies() -> None:
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    # 4 atoms in cubic fcc cell

    builder = DefectBuilder(StructureGeneratorConfig())

    # Apply 25% vacancies (1 atom out of 4)
    new_atoms = builder.apply_vacancies(atoms, 0.25)

    assert len(new_atoms) == 3
    assert len(atoms) == 4  # original is untouched


def test_apply_strain() -> None:
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    original_volume = atoms.get_volume()

    builder = DefectBuilder(StructureGeneratorConfig())

    # Apply strain
    new_atoms = builder.apply_strain(atoms, 0.15)

    new_volume = new_atoms.get_volume()

    # Volume should be different
    assert new_volume != original_volume
    # Atom count should be the same
    assert len(new_atoms) == len(atoms)


def test_apply_antisite_defects() -> None:
    # Build an ordered alloy like B2 FeAl
    atoms = bulk("Fe", "bcc", a=2.86, cubic=True)
    # Change the central atom to Al
    symbols = list(atoms.get_chemical_symbols())
    symbols[1] = "Al"  # assuming index 1 is body center in cubic
    atoms.set_chemical_symbols(symbols)

    # Make a supercell to have more atoms
    supercell = atoms.repeat((2, 2, 2))

    original_symbols = list(supercell.get_chemical_symbols())

    builder = DefectBuilder(StructureGeneratorConfig())
    new_supercell = builder.apply_antisite_defects(supercell, 0.2)

    new_symbols = list(new_supercell.get_chemical_symbols())

    # Symbols should be altered (some Fe became Al and vice versa)
    assert new_symbols != original_symbols

    # Overall composition should be exactly the same
    assert new_symbols.count("Fe") == original_symbols.count("Fe")
    assert new_symbols.count("Al") == original_symbols.count("Al")
