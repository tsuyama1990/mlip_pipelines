import logging
import random

from ase import Atoms

from src.domain_models.config import StructureGeneratorConfig


class DefectBuilder:
    """Builds defect structures (vacancies, substitutions, strain) based on exploration policies."""

    def __init__(self, config: StructureGeneratorConfig | None = None) -> None:
        if config is None:
            self.config = StructureGeneratorConfig()
        else:
            self.config = config
        self.rng = random.Random(self.config.seed_base)  # noqa: S311

    def apply_vacancies(self, atoms: Atoms, n_defects_ratio: float) -> Atoms:
        """Removes a percentage of atoms randomly to create vacancies."""
        if not (0.0 <= n_defects_ratio < 1.0):
            msg = "n_defects_ratio must be between 0.0 and 1.0 (exclusive)."
            raise ValueError(msg)

        if n_defects_ratio == 0.0 or len(atoms) == 0:
            return atoms.copy()  # type: ignore[no-untyped-call]

        n_remove = max(1, int(len(atoms) * n_defects_ratio))
        indices_to_remove = self.rng.sample(range(len(atoms)), n_remove)
        indices_to_remove.sort(reverse=True)

        new_atoms = atoms.copy()  # type: ignore[no-untyped-call]
        for idx in indices_to_remove:
            del new_atoms[idx]

        return new_atoms

    def apply_strain(self, atoms: Atoms, strain_range: float) -> Atoms:
        """Applies random volumetric and shear strain to the cell."""
        import numpy as np

        if strain_range < 0.0:
            msg = "strain_range must be non-negative."
            raise ValueError(msg)

        if strain_range == 0.0:
            return atoms.copy()  # type: ignore[no-untyped-call]

        new_atoms = atoms.copy()  # type: ignore[no-untyped-call]
        cell = new_atoms.get_cell()

        # Generate a random symmetric strain tensor
        strain_tensor = np.zeros((3, 3))
        for i in range(3):
            for j in range(i, 3):
                # Random strain between -strain_range and +strain_range
                val = self.rng.uniform(-strain_range, strain_range)
                strain_tensor[i, j] = val
                strain_tensor[j, i] = val

        deformation_gradient = np.eye(3) + strain_tensor
        new_cell = np.dot(deformation_gradient, cell)

        # Ensure the volume hasn't collapsed completely or inverted (safeguard)
        if np.linalg.det(new_cell) <= 0:
            logging.warning("Strain resulted in invalid cell volume. Returning original structure.")
            return atoms.copy()  # type: ignore[no-untyped-call]

        new_atoms.set_cell(new_cell, scale_atoms=True)  # type: ignore[no-untyped-call]

        return new_atoms

    def apply_antisite_defects(self, atoms: Atoms, n_defects_ratio: float) -> Atoms:
        """Swaps positions of different element types to create anti-site defects."""
        if not (0.0 <= n_defects_ratio <= 1.0):
            msg = "n_defects_ratio must be between 0.0 and 1.0."
            raise ValueError(msg)

        unique_symbols = list(set(atoms.get_chemical_symbols()))
        if len(unique_symbols) < 2 or n_defects_ratio == 0.0 or len(atoms) < 2:
            return atoms.copy()  # type: ignore[no-untyped-call]

        n_swaps = max(
            1, int(len(atoms) * n_defects_ratio / 2)
        )  # Div by 2 because each swap involves 2 atoms

        new_atoms = atoms.copy()  # type: ignore[no-untyped-call]
        symbols = list(new_atoms.get_chemical_symbols())

        for _ in range(n_swaps):
            idx1 = self.rng.randint(0, len(atoms) - 1)
            # Find an atom of a different type
            attempts = 0
            max_attempts = 100
            swapped = False
            while attempts < max_attempts:
                idx2 = self.rng.randint(0, len(atoms) - 1)
                if symbols[idx1] != symbols[idx2]:
                    # Swap
                    symbols[idx1], symbols[idx2] = symbols[idx2], symbols[idx1]
                    swapped = True
                    break
                attempts += 1
            if not swapped:
                msg = f"Failed to find a valid anti-site pair after {max_attempts} attempts. Structure might be too uniform or small."
                raise RuntimeError(msg)

        new_atoms.set_chemical_symbols(symbols)  # type: ignore[no-untyped-call]
        return new_atoms
