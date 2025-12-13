from typing import List, Union
import numpy as np
from ase import Atoms
from ase.cell import Cell

class BoxCarver:
    """
    Carves out a box from a larger structure for local optimization.
    """
    def __init__(self, atoms: Atoms, center_index: int, box_vector: Union[float, List[float]]):
        """
        Initialize the BoxCarver.

        Parameters
        ----------
        atoms : Atoms
            The parent structure.
        center_index : int
            Index of the atom to center the box on.
        box_vector : Union[float, List[float]]
            Size of the box. If float, it's a cube. If list, it's [lx, ly, lz].
        """
        self.atoms = atoms.copy()
        self.center_index = center_index

        if isinstance(box_vector, (float, int)):
            self.box_vector = np.array([float(box_vector)] * 3)
        else:
            self.box_vector = np.array(box_vector, dtype=float)
            if self.box_vector.shape != (3,):
                 raise ValueError("box_vector must be a float or a list of 3 floats.")

    def carve(self, cluster_mode: bool = True) -> Atoms:
        """
        Perform the carving operation.

        Parameters
        ----------
        cluster_mode : bool, optional
            If True, creates a non-periodic cluster with vacuum.
            If False, creates a periodic box with the carve size.
            Default is True.

        Returns
        -------
        Atoms
            The carved structure.
        """
        # 1. Centering: Shift atoms so center_index is at (0.5, 0.5, 0.5) in scaled coordinates
        # We work with a copy to avoid modifying the original during calculation
        work_atoms = self.atoms.copy()

        # Get scaled positions
        scaled_positions = work_atoms.get_scaled_positions()
        center_scaled_pos = scaled_positions[self.center_index]

        # Calculate shift vector to move center atom to (0.5, 0.5, 0.5)
        shift = np.array([0.5, 0.5, 0.5]) - center_scaled_pos

        # Apply shift and wrap
        scaled_positions = (scaled_positions + shift) % 1.0
        work_atoms.set_scaled_positions(scaled_positions)

        # 2. Cutout: Extract atoms within +/- box_vector / 2 from the center
        # Convert back to Cartesian for distance check.
        # Note: The cell is periodic, so "wrapping" handled by set_scaled_positions is good.
        # But we need to be careful about the "center" in Cartesian.
        # The center is now at 0.5 * cell_lengths (approx, if orthogonal).
        # Better: get the cartesian position of the center atom (now at 0.5, 0.5, 0.5)

        # To handle non-orthogonal cells correctly, we should define the cutout region
        # based on Cartesian difference from the center atom.
        # However, we must account for PBC if the box size is comparable to cell size.
        # The spec says: "Shift (Wrap)" -> "Simple simplified PBC crossing".
        # So we can assume after wrapping, we just take Cartesian distance from the new center.

        center_cart_pos = work_atoms.positions[self.center_index]

        # We need to find atoms where |r_i - r_center| < box/2 in each dimension.
        # Since we centered and wrapped, the "image" we want is the one closest to the center.
        # But wait, we just wrapped everything into [0, 1].
        # So simple Cartesian difference might fail if the box is large and wraps again?
        # The spec implies we carve *from the shifted view*.
        # So we just check: is position within [center - box/2, center + box/2]?

        positions = work_atoms.get_positions()
        # center_cart_pos should be computed from the wrapped positions
        center_cart_pos = positions[self.center_index]

        lower_bound = center_cart_pos - self.box_vector / 2.0
        upper_bound = center_cart_pos + self.box_vector / 2.0

        mask = np.all((positions >= lower_bound) & (positions <= upper_bound), axis=1)
        indices = np.where(mask)[0]

        if len(indices) <= 1:
            # "Validation: ValueError if atoms count is extremely low (e.g. 1)"
            # 0 or 1 atoms is too few.
            raise ValueError(f"Carved box contains too few atoms ({len(indices)}).")

        carved_atoms = work_atoms[indices]

        # 3. Re-wrapping
        if cluster_mode:
            # Cluster Mode: Non-periodic, vacuum=10.0A
            carved_atoms.pbc = False
            carved_atoms.center(vacuum=10.0)
        else:
            # Periodic Mode: Cell size = box size
            # We need to center the atoms in the new box.
            # The carved atoms are currently centered around `center_cart_pos`.
            # We want them in a box of size `self.box_vector`.
            # Let's shift them so the center atom is at `self.box_vector / 2`.

            new_center = self.box_vector / 2.0
            offset = new_center - center_cart_pos
            carved_atoms.positions += offset

            carved_atoms.set_cell(self.box_vector)
            carved_atoms.pbc = True

        return carved_atoms
