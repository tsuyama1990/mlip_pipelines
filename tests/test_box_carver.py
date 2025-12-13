import numpy as np
import pytest
from ase import Atoms
from src.carvers.box_carver import BoxCarver

def test_box_carver_simple():
    """Test basic carving functionality."""
    # Create a simple cubic crystal
    atoms = Atoms('Cu4',
                  positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]],
                  cell=[4, 4, 4],
                  pbc=True)

    # Center is atom 0 at [0,0,0]
    # Box size 3.0 -> should capture all atoms since they are at dist 2.0 (<= 1.5 is false)
    # Wait, box size 3.0 means +/- 1.5. Atom at 2.0 is outside.
    # So only center atom should remain?
    # Spec says: ValueError if atoms count is <= 1.

    # Let's try box size 5.0 -> should capture all
    carver = BoxCarver(atoms, center_index=0, box_vector=5.0)
    carved = carver.carve(cluster_mode=True)

    assert len(carved) == 4
    assert not carved.pbc.any() # Cluster mode
    assert carved.cell[0][0] > 10.0 # Vacuum added

def test_box_carver_pbc_wrapping():
    """Test wrapping across boundaries."""
    # Atom at 0.1 and 9.9 in a 10.0 box are close (dist 0.2).
    atoms = Atoms('H2',
                  positions=[[0.1, 0.1, 0.1], [9.9, 9.9, 9.9]],
                  cell=[10, 10, 10],
                  pbc=True)

    # Carve around atom 0 (at 0.1).
    # Box size 2.0 (radius 1.0).
    # Should include atom 1 (at 9.9 -> -0.1 relative to 0.1? No, 0.1 is center.
    # Wrapped coordinates:
    # 0.1 -> shifted to 0.5 (shift = 0.4).
    # 9.9 -> 9.9 + 0.4 = 10.3 -> 0.3 (scaled).
    # 0.5 is center. 0.3 is dist 0.2 away.
    # Box size 2.0 in 10.0 cell is 0.2 in scaled.
    # +/- 0.1 scaled.
    # 0.3 is at 0.2 distance from 0.5.
    # Wait.
    # Center at 0.5. Box size 2.0 corresponds to +/- 1.0.
    # Scaled box size = 2.0/10.0 = 0.2.
    # Range: [0.5 - 0.1, 0.5 + 0.1] = [0.4, 0.6].
    # Atom 1 is at 9.9.
    # Shifted: 9.9 + (5.0 - 0.1) = 9.9 + 4.9 = 14.8 -> 4.8?

    # Let's trace logic.
    # Center atom: [0.1]. Scaled: 0.01.
    # Shift: 0.5 - 0.01 = 0.49.
    # Atom 1: [9.9]. Scaled: 0.99.
    # Apply shift: 0.99 + 0.49 = 1.48 -> 0.48.
    # Center is at 0.50.
    # Atom 1 is at 0.48. Dist = 0.02 scaled -> 0.2 Angstrom.
    # Box size 2.0. Radius 1.0.
    # 0.2 < 1.0. Should be included.

    carver = BoxCarver(atoms, center_index=0, box_vector=2.0)
    carved = carver.carve()

    assert len(carved) == 2

def test_validation_error():
    atoms = Atoms('H', positions=[[0,0,0]], cell=[10,10,10], pbc=True)
    carver = BoxCarver(atoms, center_index=0, box_vector=1.0)
    with pytest.raises(ValueError):
        carver.carve()
