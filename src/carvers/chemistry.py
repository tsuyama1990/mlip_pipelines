from typing import Tuple, List, Optional
import numpy as np
from ase import Atoms
from collections import Counter
from loguru import logger

try:
    from pymatgen.core import Composition
except ImportError:
    Composition = None
    logger.warning("pymatgen not found. StoichiometryGuard will fail if instantiated.")

class StoichiometryGuard:
    """
    Guards the stoichiometry of a cluster to match a target formula.
    """
    def __init__(self, target_formula: str):
        if Composition is None:
            raise ImportError("pymatgen is required for StoichiometryGuard.")

        self.target_formula = target_formula
        self.comp = Composition(target_formula)
        # Normalize to simplest ratio
        self.reduced_comp, _ = self.comp.get_reduced_composition_and_factor()
        self.formula_size = self.reduced_comp.num_atoms

        # Calculate target fractions
        self.target_fractions = {
            str(el): amt / self.formula_size
            for el, amt in self.reduced_comp.as_dict().items()
        }

    def check(self, atoms: Atoms, tolerance: float = 0.1) -> bool:
        """
        Check if the atoms object matches the target stoichiometry within tolerance.
        """
        syms = atoms.get_chemical_symbols()
        total = len(syms)
        if total == 0:
            return False

        counts = Counter(syms)

        for el, target_frac in self.target_fractions.items():
            current_frac = counts.get(el, 0) / total
            if abs(current_frac - target_frac) > tolerance:
                return False

        # Check for alien elements
        for el in counts:
            if el not in self.target_fractions:
                return False

        return True

    def correct(self, atoms: Atoms, center_index: int = None) -> Tuple[Atoms, List[int]]:
        """
        Correct the stoichiometry by removing excess atoms furthest from the center.

        Returns
        -------
        Atoms
            The corrected atoms object.
        List[int]
            The indices of the atoms in the original input object that were kept.
        """
        syms = atoms.get_chemical_symbols()
        total = len(syms)
        counts = Counter(syms)

        # Calculate max integer formula units we can form
        num_units = float('inf')
        for el, amt in self.reduced_comp.as_dict().items():
            count = counts.get(str(el), 0)
            possible_units = count / amt
            if possible_units < num_units:
                num_units = possible_units

        target_units = int(np.floor(num_units))

        if target_units == 0:
            logger.warning(f"Stoichiometry correction for {self.target_formula} would deplete cluster. Returning original.")
            return atoms, list(range(len(atoms)))

        # Target counts for each element
        target_counts = {
            str(el): int(amt * target_units)
            for el, amt in self.reduced_comp.as_dict().items()
        }

        # Determine center position
        if center_index is not None and 0 <= center_index < len(atoms):
            center_pos = atoms.positions[center_index]
        else:
            center_pos = np.mean(atoms.get_positions(), axis=0)

        positions = atoms.get_positions()
        dist_sq = np.sum((positions - center_pos)**2, axis=1)

        indices_to_remove = set()

        # Identify excess for each element
        for el in counts:
            current_count = counts[el]
            target_count = target_counts.get(el, 0)
            excess = current_count - target_count

            if excess > 0:
                # Get indices of this element
                el_indices = [i for i, s in enumerate(syms) if s == el]
                # Sort by distance descending (furthest first)
                el_indices.sort(key=lambda i: dist_sq[i], reverse=True)
                # Mark for removal
                indices_to_remove.update(el_indices[:excess])

        # Check for elements not in target
        for i, s in enumerate(syms):
            if s not in target_counts and i not in indices_to_remove:
                indices_to_remove.add(i)

        if not indices_to_remove:
            return atoms, list(range(len(atoms)))

        # Create new atoms and list of kept indices
        mask = np.ones(len(atoms), dtype=bool)
        mask[list(indices_to_remove)] = False

        kept_indices = np.where(mask)[0].tolist()
        corrected_atoms = atoms[mask]

        logger.info(f"StoichiometryGuard: Removed {len(indices_to_remove)} atoms to match {self.target_formula}.")
        return corrected_atoms, kept_indices
