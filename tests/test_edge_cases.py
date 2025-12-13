import pytest
import numpy as np
from ase import Atoms
from src.core.validators.structure_validator import validate_structure, validate_no_atomic_clash, InvalidStructureError

class TestEdgeCases:
    def test_empty_system(self):
        """MASD 4.2: Handle N=0"""
        atoms = Atoms()
        with pytest.raises(InvalidStructureError, match="Empty structure"):
            validate_structure(atoms)

    def test_single_atom(self):
        """MASD 4.2: Handle N=1"""
        atoms = Atoms('H', positions=[[0, 0, 0]])
        validate_no_atomic_clash(atoms)  # Should not crash (no pairs)

    def test_nan_detection(self):
        """MASD 4.3: Detect NaN in coordinates"""
        atoms = Atoms('Si2', positions=[[0, 0, 0], [np.nan, 0, 0]])
        with pytest.raises(InvalidStructureError, match="NaN"):
            validate_structure(atoms)

    def test_inf_detection(self):
        """MASD 4.3: Detect Inf in coordinates"""
        atoms = Atoms('Si2', positions=[[0, 0, 0], [np.inf, 0, 0]])
        with pytest.raises(InvalidStructureError, match="Inf"):
            validate_structure(atoms)
