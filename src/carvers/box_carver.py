from typing import List, Union, Optional
import numpy as np
from ase import Atoms
from ase.cell import Cell
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from loguru import logger
from ase.calculators.calculator import Calculator
import tempfile
from carvers.chemistry import StoichiometryGuard

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

        Raises
        ------
        ValueError
            If box_vector is not valid.
        """
        self.atoms = atoms.copy()
        self.center_index = center_index

        if isinstance(box_vector, (float, int)):
            self.box_vector = np.array([float(box_vector)] * 3)
        else:
            self.box_vector = np.array(box_vector, dtype=float)
            if self.box_vector.shape != (3,):
                 raise ValueError("box_vector must be a float or a list of 3 floats.")

    def carve(self, cluster_mode: bool = True, skin_depth: int = 1, calculator: Optional[Calculator] = None, stoichiometry: Optional[str] = None) -> Atoms:
        """
        Perform the carving operation with Smart Carving logic.

        Parameters
        ----------
        cluster_mode : bool, optional
            If True, creates a vacuum cluster. Default is True.
        skin_depth : int, optional
            Number of neighbor layers to expand connectivity. Default is 1.
        calculator : Calculator, optional
            ASE calculator for pre-relaxation. Default is None.
        stoichiometry : str, optional
            Target stoichiometry formula (e.g., "NaCl") to enforce.

        Returns
        -------
        Atoms
            The carved structure.

        Raises
        ------
        ValueError
            If the carved box contains too few atoms.
        """
        work_atoms = self.atoms.copy()

        # 1. Centering & Wrapping
        self._center_and_wrap(work_atoms)

        # Calculate center position in the wrapped frame
        center_cart_pos = work_atoms.positions[self.center_index]

        # 2. Geometric Cutout & Healing
        initial_indices = self._geometric_cut(work_atoms, center_cart_pos)
        final_indices = self._heal(work_atoms, initial_indices, skin_depth)

        if len(final_indices) <= 1:
             raise ValueError(f"Carved box contains too few atoms ({len(final_indices)}).")

        final_indices.sort()
        # Create intermediate cluster
        carved_atoms = work_atoms[final_indices]

        # 2.5 Stoichiometry Correction
        if stoichiometry:
            try:
                guard = StoichiometryGuard(stoichiometry)

                # Determine center index in the local cluster
                try:
                    center_local_idx = final_indices.index(self.center_index)
                except ValueError:
                    center_local_idx = None

                corrected_atoms, kept_local_indices = guard.correct(carved_atoms, center_index=center_local_idx)

                # Update final_indices based on what was kept
                # kept_local_indices refers to indices in 'carved_atoms' (which corresponds to final_indices)
                new_final_indices = [final_indices[i] for i in kept_local_indices]
                final_indices = new_final_indices
                final_indices.sort()

                carved_atoms = corrected_atoms

            except Exception as e:
                logger.warning(f"Stoichiometry correction failed or skipped: {e}")

        self._check_stoichiometry(carved_atoms)

        # 3. Re-wrapping / Cluster Setup
        # Note: We must work with a new Atoms object constructed from final_indices of work_atoms
        # because _finalize_cell modifies positions/cell.
        # But wait, carved_atoms IS that object.
        # However, _finalize_cell expects the atoms to be in the frame of work_atoms (for periodic)
        # or relative to center.
        # If we modified carved_atoms in guard, we removed atoms. The positions are preserved.

        self._finalize_cell(carved_atoms, cluster_mode, center_cart_pos)

        # 4. Pre-relaxation
        if calculator is not None:
            self._pre_relax(carved_atoms, calculator, set(initial_indices), final_indices)

        return carved_atoms

    def _center_and_wrap(self, atoms: Atoms) -> None:
        """
        Shift and wrap atoms so the center atom is at (0.5, 0.5, 0.5).
        Respects partial PBC.
        """
        scaled_positions = atoms.get_scaled_positions()
        center_scaled_pos = scaled_positions[self.center_index]
        shift = np.array([0.5, 0.5, 0.5]) - center_scaled_pos
        new_scaled_positions = scaled_positions + shift

        pbc = atoms.pbc
        for i in range(3):
            if pbc[i]:
                new_scaled_positions[:, i] = new_scaled_positions[:, i] % 1.0

        atoms.set_scaled_positions(new_scaled_positions)

    def _geometric_cut(self, atoms: Atoms, center_cart_pos: np.ndarray) -> set:
        """
        Identify atoms within the geometric box around the center.
        """
        positions = atoms.get_positions()

        lower_bound = center_cart_pos - self.box_vector / 2.0
        upper_bound = center_cart_pos + self.box_vector / 2.0

        mask = np.all((positions >= lower_bound) & (positions <= upper_bound), axis=1)
        return set(np.where(mask)[0])

    def _heal(self, atoms: Atoms, initial_indices: set, skin_depth: int) -> List[int]:
        """
        Expand the initial selection by including neighbors (healing).
        """
        if skin_depth <= 0:
            return list(initial_indices)

        cutoffs = natural_cutoffs(atoms, mult=1.2)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)

        current_indices = initial_indices.copy()

        for _ in range(skin_depth):
            new_neighbors = set()
            for idx in current_indices:
                indices, offsets = nl.get_neighbors(idx)
                for neighbor_idx in indices:
                    if neighbor_idx not in current_indices:
                        new_neighbors.add(neighbor_idx)
            current_indices.update(new_neighbors)

        return list(current_indices)

    def _finalize_cell(self, atoms: Atoms, cluster_mode: bool, center_cart_pos: np.ndarray) -> None:
        """
        Set up the cell and PBC for the carved atoms.
        """
        if cluster_mode:
            atoms.pbc = False
            atoms.center(vacuum=10.0)
        else:
            # Periodic Mode: center atoms in the NEW box.
            new_center = self.box_vector / 2.0
            offset = new_center - center_cart_pos
            atoms.positions += offset
            atoms.set_cell(self.box_vector)
            atoms.pbc = True

    def _check_stoichiometry(self, cluster: Atoms) -> None:
        """Check if cluster stoichiometry deviates from bulk."""
        bulk_syms = self.atoms.get_chemical_symbols()
        cluster_syms = cluster.get_chemical_symbols()

        from collections import Counter
        bulk_counts = Counter(bulk_syms)
        cluster_counts = Counter(cluster_syms)

        bulk_total = len(bulk_syms)
        cluster_total = len(cluster_syms)

        if cluster_total == 0:
            return

        for elem, count in bulk_counts.items():
            bulk_ratio = count / bulk_total
            cluster_ratio = cluster_counts.get(elem, 0) / cluster_total

            if abs(bulk_ratio - cluster_ratio) > 0.15:
                logger.warning(f"Stoichiometry mismatch for {elem}: Bulk={bulk_ratio:.2f}, Cluster={cluster_ratio:.2f}")

    def _pre_relax(self, cluster: Atoms, calculator: Calculator, initial_indices_set: set, final_indices_list: List[int]) -> None:
        """
        Relax the cluster with fixed boundary.
        """
        cluster.calc = calculator

        fixed_indices = []
        for i, original_idx in enumerate(final_indices_list):
            if original_idx not in initial_indices_set:
                fixed_indices.append(i)

        if fixed_indices:
            c = FixAtoms(indices=fixed_indices)
            cluster.set_constraint(c)

        with tempfile.NamedTemporaryFile(mode='w', delete=True) as tmp:
            try:
                dyn = BFGS(cluster, logfile=tmp.name)
                dyn.run(fmax=0.5, steps=10)
            except Exception as e:
                logger.warning(f"Pre-relaxation failed: {e}")

        cluster.set_constraint()
