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

        Handles partial PBC correctly by only wrapping dimensions where pbc is True.
        """
        work_atoms = self.atoms.copy()

        # Check box size validity relative to cell (warn or error?)
        # If box > cell, we might get empty results or need supercells.
        # For this implementation, we assume the user knows what they are doing,
        # but we guard against completely invalid states.
        cell_lengths = work_atoms.cell.lengths()
        if np.any((self.box_vector > cell_lengths) & work_atoms.pbc):
            # If box is larger than cell in a periodic dim, centering approach is ambiguous without supercell.
            # But "Carver" usually implies cutting a piece out.
            # We proceed, but the result might be the whole cell (or close to it).
            pass

        # 1. Centering
        scaled_positions = work_atoms.get_scaled_positions()
        center_scaled_pos = scaled_positions[self.center_index]

        # Shift vector: move center to 0.5
        shift = np.array([0.5, 0.5, 0.5]) - center_scaled_pos

        # Apply shift
        new_scaled_positions = scaled_positions + shift

        # Wrap ONLY if PBC is True for that dimension
        pbc = work_atoms.pbc
        for i in range(3):
            if pbc[i]:
                new_scaled_positions[:, i] = new_scaled_positions[:, i] % 1.0
            else:
                # For non-periodic dims, we don't wrap.
                # But we shifted everyone.
                # If the center was at 0.1, we added 0.4.
                # If an atom was at 0.9, it became 1.3.
                # This preserves relative distances, which is what matters.
                pass

        work_atoms.set_scaled_positions(new_scaled_positions)

        # 2. Cutout
        positions = work_atoms.get_positions()
        center_cart_pos = positions[self.center_index]

        lower_bound = center_cart_pos - self.box_vector / 2.0
        upper_bound = center_cart_pos + self.box_vector / 2.0

        # Check boundaries
        mask = np.all((positions >= lower_bound) & (positions <= upper_bound), axis=1)
        indices = np.where(mask)[0]

        if len(indices) <= 1:
            raise ValueError(f"Carved box contains too few atoms ({len(indices)}).")

        carved_atoms = work_atoms[indices]

        # 3. Re-wrapping
        if cluster_mode:
            carved_atoms.pbc = False
            carved_atoms.center(vacuum=10.0)
        else:
            new_center = self.box_vector / 2.0
            offset = new_center - center_cart_pos
            carved_atoms.positions += offset

            carved_atoms.set_cell(self.box_vector)
            carved_atoms.pbc = True

        return carved_atoms
