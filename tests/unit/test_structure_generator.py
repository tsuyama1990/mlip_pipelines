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

    candidates = list(generator.generate_local_candidates(mod_atoms, n=50))
    assert len(list(candidates)) == 5


def test_generate_local_candidates_normal():
    """Test normal candidate generation."""
    config = StructureGeneratorConfig()
    generator = StructureGenerator(config)

    atoms = Atoms("Fe", positions=[(0, 0, 0)])
    atoms.set_cell([10, 10, 10])

    candidates = list(generator.generate_local_candidates(atoms, n=5))
    assert len(list(candidates)) == 5

    # Rattling should move the atom
    for cand in list(candidates):
        assert cand.positions[0][0] != 0.0 or cand.positions[0][1] != 0.0


def test_generate_interface_invalid():
    """Test interface generation with invalid elements."""
    config = StructureGeneratorConfig()
    generator = StructureGenerator(config)

    target = InterfaceTarget(element1="Unobtainium", element2="Fe")
    import pytest

    with pytest.raises(ValueError, match="Unsupported or disallowed interface element"):
        generator.generate_interface(target)


def test_extract_intelligent_cluster() -> None:
    from ase.build import bulk

    from src.domain_models.config import CutoutConfig

    config = StructureGeneratorConfig(seed_base=1)
    gen = StructureGenerator(config)

    # 3x3x3 bulk Fe block (27 atoms)
    system = bulk("Fe", crystalstructure="bcc", a=2.86, cubic=True) * (3, 3, 3)

    cutout_config = CutoutConfig(
        core_radius=2.0,
        buffer_radius=4.0,
        enable_pre_relaxation=False,
        enable_passivation=False
    )

    # Target the center atom
    target_idx = [13]

    cluster = gen.extract_intelligent_cluster(system, target_idx, cutout_config)

    assert len(cluster) < len(system)
    assert len(cluster) > 1

    # Check force weights
    assert "force_weights" in cluster.arrays
    fw = cluster.arrays["force_weights"]

    # Target atom and very close neighbors should have weight 1.0
    core_atoms = sum(1 for w in fw if w == 1.0)
    assert core_atoms >= 1


def test_hydrogen_passivation() -> None:
    from ase.build import bulk

    from src.domain_models.config import CutoutConfig

    config = StructureGeneratorConfig(seed_base=1)
    gen = StructureGenerator(config)

    # Very small cluster to test passivation
    cluster = bulk("MgO", crystalstructure="rocksalt", a=4.21) * (1, 1, 1)

    # Need a bigger cluster to test passivation, a 1x1x1 only has 2 atoms, so it's fully exposed
    # but natural_cutoffs might not trigger properly if neighbors don't match our simple heuristic
    cluster = bulk("MgO", crystalstructure="rocksalt", a=4.21) * (2, 2, 2)

    # Manually remove an atom to create an under-coordinated surface
    del cluster[10]

    cutout_config = CutoutConfig(
        core_radius=1.0,
        buffer_radius=5.0,
        enable_pre_relaxation=False,
        enable_passivation=True,
        passivation_element="H"
    )

    # Use a target closer to the removed atom
    target_idx = [9]
    passivated_cluster = gen.extract_intelligent_cluster(cluster, target_idx, cutout_config)

    assert "H" in passivated_cluster.symbols
    # Check force_weights correctly applied to newly added passivating H atoms
    assert len(passivated_cluster.arrays["force_weights"]) == len(passivated_cluster)


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
