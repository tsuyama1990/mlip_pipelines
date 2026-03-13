from typing import Any

import numpy as np

from src.core.interfaces import AbstractOracle
from src.domain_models.config import OracleConfig


class DFTOracle(AbstractOracle):
    def __init__(self, config: OracleConfig) -> None:
        self.config = config

    def compute(self, structures: list[Any]) -> list[Any]:
        """Compute exact properties using a dummy potential for Cycle 1 verification."""
        computed_structures = []
        for atoms in structures:
            atoms_cp = atoms.copy()
            # Dummy logic mimicking DFT forces
            forces = np.random.randn(len(atoms_cp), 3) * 0.1
            energy = np.sum(np.random.randn(len(atoms_cp)))

            # Since we can't easily attach a mock calculator without breaking rules, we'll
            # attach arrays directly to the atoms object so the trainer can extract them.
            atoms_cp.arrays["forces"] = forces
            atoms_cp.info["energy"] = energy

            computed_structures.append(atoms_cp)
        return computed_structures
