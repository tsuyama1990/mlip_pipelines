import numpy as np
import pytest
from ase import Atoms
from src.carvers.box_carver import BoxCarver
from ase.calculators.singlepoint import SinglePointCalculator
from loguru import logger
try:
    from ase.calculators.emt import EMT
except ImportError:
    EMT = None

def test_box_carver_simple():
    """Test basic carving functionality."""
    atoms = Atoms('Cu4',
                  positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]],
                  cell=[4, 4, 4],
                  pbc=True)

    carver = BoxCarver(atoms, center_index=0, box_vector=5.0)
    carved = carver.carve(cluster_mode=True, skin_depth=0)

    assert len(carved) == 4
    assert not carved.pbc.any() # Cluster mode

def test_box_carver_healing():
    """Test connectivity expansion (healing)."""
    # Create a trimer Cu-Cu-Cu at dist 2.0.
    # 0 at 0.0, 1 at 2.0, 2 at 4.0.
    # Box size 3.0 around atom 0.
    # Geometric cut [-1.5, 1.5] includes only atom 0.
    # This triggers ValueError (too few atoms).
    # Let's shift so 0 and 1 are inside.
    # Center at 1.0 (midpoint of 0-1).
    # But center_index must be an integer (an atom).
    # Let's use 2 atoms inside, 1 outside.
    # Atom 0 at 0.0, Atom 1 at 1.0, Atom 2 at 3.0 (dist 2.0 from 1).
    atoms = Atoms('Cu3', positions=[[0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]], cell=[10, 10, 10], pbc=True)

    carver = BoxCarver(atoms, center_index=0, box_vector=2.5)
    # Center atom 0. Box [-1.25, 1.25].
    # Atom 0 (0.0) -> Inside.
    # Atom 1 (1.0) -> Inside.
    # Atom 2 (3.0) -> Outside.

    # Without skin: Should get 0 and 1.
    carved_no_skin = carver.carve(skin_depth=0)
    assert len(carved_no_skin) == 2

    # With skin: Should pull atom 2 (neighbor of 1).
    # Dist(1, 2) = 2.0.
    # Cu covalent radius is ~1.3, so bond ~2.6.
    # natural_cutoffs(mult=1.2) should catch 2.0.
    carved_skin = carver.carve(skin_depth=1)
    assert len(carved_skin) == 3

def test_pre_relaxation():
    """Test pre-relaxation moves atoms and fixes boundary."""
    if EMT is None:
        pytest.skip("EMT calculator not available")

    calc = EMT()

    # Setup: A-B-C chain. A, B inside. C outside but pulled by healing.
    # A=0.0, B=2.4, C=4.8.
    atoms = Atoms('Cu3', positions=[[0, 0, 0], [2.4, 0, 0], [4.8, 0, 0]], cell=[20, 20, 20], pbc=True)

    # Box size 6.0 around A(0). Range [-3.0, 3.0].
    # A(0) inside. B(2.4) inside. C(4.8) outside.
    carver = BoxCarver(atoms, center_index=0, box_vector=6.0)

    # Verify skin behavior first
    carved = carver.carve(calculator=None, skin_depth=1)
    assert len(carved) == 3 # C pulled in

    # Now with relaxation
    # C is the "skin" (added by healing). It should be fixed.
    # A and B should relax.
    carved_relaxed = carver.carve(calculator=calc, skin_depth=1)

    # Let's assert positions changed.
    # Initial relative pos: 0, 2.4, 4.8.
    # If relaxed, they should change.
    d01 = carved_relaxed.get_distance(0, 1)
    d12 = carved_relaxed.get_distance(1, 2)

    assert abs(d01 - 2.4) > 0.01 or abs(d12 - 2.4) > 0.01

def test_stoichiometry_warning():
    """Test stoichiometry warning."""
    # Bulk: NaCl (1:1)
    # Create enough atoms to avoid "too few" error.
    # 2 units of NaCl.
    atoms = Atoms('Na2Cl2', positions=[[0,0,0], [2,0,0], [0,2,0], [2,2,0]], cell=[10,10,10], pbc=True)
    # 0:Na, 1:Cl, 2:Na, 3:Cl (default order depends on chemical symbols string parsing, usually Na, Na, Cl, Cl?)
    # ase: 'Na2Cl2' -> Na, Na, Cl, Cl.
    atoms.set_chemical_symbols(['Na', 'Cl', 'Na', 'Cl'])

    atoms = Atoms('Na2Cl2', positions=[[0,0,0], [1,0,0], [5,0,0], [6,0,0]], cell=[10,10,10], pbc=True)
    # 0:Na, 1:Na, 2:Cl, 3:Cl.

    carver = BoxCarver(atoms, center_index=0, box_vector=2.0)
    # Should get 0 and 1 (Na, Na).
    # Ratio Na=1.0, Cl=0.0.
    # Bulk: Na=0.5, Cl=0.5.
    # Mismatch!

    logs = []
    # Capture loguru logs
    logger.add(lambda msg: logs.append(msg))

    carver.carve(skin_depth=0)

    assert any("Stoichiometry mismatch" in str(msg) for msg in logs)

def test_validation_error():
    atoms = Atoms('H', positions=[[0,0,0]], cell=[10,10,10], pbc=True)
    carver = BoxCarver(atoms, center_index=0, box_vector=1.0)
    with pytest.raises(ValueError):
        carver.carve()
