import pytest
import numpy as np
from ase import Atoms
from ase.build import bulk
from core.validators.structure_validator import (
    validate_no_atomic_clash,
    validate_cell,
    validate_structure,
    InvalidStructureError
)

class TestAtomicClashDetection:
    def test_reject_close_atoms(self):
        """MASD 1.2: Reject atoms closer than 0.6 * covalent_radii_sum"""
        # H covalent radius ~0.31 A. Sum ~0.62. 0.6 * 0.62 = 0.372 A.
        # 0.01 A is definitely too close.
        atoms = Atoms('H2', positions=[[0, 0, 0], [0.01, 0, 0]])
        with pytest.raises(InvalidStructureError, match="Atomic clash"):
            validate_no_atomic_clash(atoms)

    def test_accept_normal_distances(self):
        """Accept physically reasonable structures"""
        atoms = bulk('Si', cubic=True)
        validate_no_atomic_clash(atoms)  # Should not raise

    def test_pbc_minimum_image(self):
        """MASD 1.2: Verify MIC used for PBC structures"""
        atoms = bulk('Si', cubic=True)
        atoms.set_pbc([True, True, True])
        # Place atom near cell boundary
        # Cell is typically 5.43 for Si
        # Place at 0.1 and 5.33 (dist 0.2 < 0.37)
        atoms.positions[0] = [0.1, 0.1, 0.1]
        atoms.positions[1] = [5.33, 5.33, 5.33]  # Wrapped distance is small

        # Verify it clashes
        with pytest.raises(InvalidStructureError, match="Atomic clash"):
            validate_no_atomic_clash(atoms)

class TestCellValidation:
    def test_reject_singular_matrix(self):
        """MASD 2.1: Reject cells with det(cell) ≈ 0"""
        atoms = Atoms('Si', cell=[[1, 0, 0], [1, 0, 0], [0, 0, 1]], pbc=True)
        with pytest.raises(InvalidStructureError, match="Degenerate cell"):
            validate_cell(atoms)

    def test_extreme_aspect_ratio(self):
        """MASD 4.1: Handle needle-shaped cells"""
        atoms = Atoms('Si', cell=[1000, 1, 1], pbc=True, positions=[[0, 0, 0]])
        validate_cell(atoms)  # Should not crash and volume > 1e-6 (1000 > 1e-6)
