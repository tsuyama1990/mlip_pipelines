from ase import Atoms

from src.domain_models.config import InterfaceTarget, StructureGeneratorConfig
from src.generators.structure_generator import StructureGenerator


def test_generate_interface() -> None:
    config = StructureGeneratorConfig()
    generator = StructureGenerator(config)
    target = InterfaceTarget(element1="FePt", element2="MgO", face1="Fe", face2="Mg")

    interface = generator.generate_interface(target)

    assert isinstance(interface, Atoms)
    assert len(interface) > 0
    symbols = interface.get_chemical_symbols()
    assert "Fe" in symbols or "Pt" in symbols
    assert "Mg" in symbols or "O" in symbols


def test_generate_local_candidates_oom_protection():
    """Test OOM protection blocks large structures."""
    config = StructureGeneratorConfig()
    generator = StructureGenerator(config)

    # Too large
    huge_atoms = Atoms("Fe" * 10001)
    import pytest

    with pytest.raises(ValueError, match="too large for rattling"):
        generator.generate_local_candidates(huge_atoms)


def test_generate_local_candidates_scaling():
    """Test candidate count scaling for moderately large structures."""
    config = StructureGeneratorConfig()
    generator = StructureGenerator(config)

    # 2000 atoms, requests 50 candidates
    # Actual n should be max(1, 50 // 10) = 5
    mod_atoms = Atoms("Fe" * 2000, positions=[(0, 0, 0)] * 2000)
    mod_atoms.set_cell([10, 10, 10])

    candidates = generator.generate_local_candidates(mod_atoms, n=50)
    assert len(candidates) == 5


def test_generate_local_candidates_normal():
    """Test normal candidate generation."""
    config = StructureGeneratorConfig()
    generator = StructureGenerator(config)

    atoms = Atoms("Fe", positions=[(0, 0, 0)])
    atoms.set_cell([10, 10, 10])

    candidates = generator.generate_local_candidates(atoms, n=5)
    assert len(candidates) == 5

    # Rattling should move the atom
    for cand in candidates:
        assert cand.positions[0][0] != 0.0 or cand.positions[0][1] != 0.0


def test_generate_interface_invalid():
    """Test interface generation with invalid elements."""
    config = StructureGeneratorConfig()
    generator = StructureGenerator(config)

    target = InterfaceTarget(element1="Unobtainium", element2="Fe")
    import pytest

    with pytest.raises(ValueError, match="Invalid or unsupported element target"):
        generator.generate_interface(target)


def test_generate_interface_valid():
    """Test interface generation with valid elements."""
    config = StructureGeneratorConfig()
    generator = StructureGenerator(config)

    target = InterfaceTarget(element1="FePt", element2="MgO")

    interface_atoms = generator.generate_interface(target)

    symbols = interface_atoms.get_chemical_symbols()
    assert "Fe" in symbols
    assert "Pt" in symbols
    assert "Mg" in symbols
    assert "O" in symbols

    # Check that it's a stacked structure (more than just the bulk atoms)
    assert len(interface_atoms) > 4
