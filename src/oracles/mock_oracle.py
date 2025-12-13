import time
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from src.core.interfaces import AbstractOracle

class MockOracle(AbstractOracle):
    """
    Mock Oracle for testing and dry-runs.
    """
    def compute(self, atoms: Atoms) -> Atoms:
        """
        Perform mock DFT calculation.
        """
        # Simulate computation time
        time.sleep(0.1)

        n_atoms = len(atoms)

        # Energy: E = -4.0 * N + Noise
        energy = -4.0 * n_atoms + np.random.normal(0, 0.1)

        # Forces: Uniform(-0.1, 0.1)
        forces = np.random.uniform(-0.1, 0.1, size=(n_atoms, 3))

        # Stress: Zeros (3, 3)
        # SinglePointCalculator expects 6-element Voigt or 3x3.
        # Let's use 3x3 for simplicity if supported, or Voigt if standard ASE prefers.
        # ASE standard is usually (6,) for stress, but 3x3 is often accepted.
        # Let's return (3, 3) as per spec, but ASE calc often stores it flattened or Voigt.
        # We will pass what ASE expects. Usually SinglePointCalculator handles 'stress' as (6,) or (3,3).
        # We'll stick to (3,3) as requested in "interfaces.py docstring" -> "Stress should be (3, 3) matrix or Voigt notation (6,)."
        # But wait, SinglePointCalculator might be picky. Let's provide 3x3.
        stress = np.zeros((3, 3))

        # Create a copy to return
        atoms_calc = atoms.copy()

        calc = SinglePointCalculator(
            atoms_calc,
            energy=energy,
            forces=forces,
            stress=stress
        )
        atoms_calc.calc = calc

        return atoms_calc
