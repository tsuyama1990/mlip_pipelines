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
import os

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

    def carve(self, cluster_mode: bool = True, skin_depth: int = 1, calculator: Optional[Calculator] = None) -> Atoms:
        """
        Perform the carving operation with Smart Carving logic.

        Parameters
        ----------
        cluster_mode : bool
            If True, creates a vacuum cluster.
        skin_depth : int
            Number of neighbor layers to expand connectivity.
        calculator : Calculator, optional
            ASE calculator for pre-relaxation.
        """
        work_atoms = self.atoms.copy()

        # 1. Centering & Wrapping
        scaled_positions = work_atoms.get_scaled_positions()
        center_scaled_pos = scaled_positions[self.center_index]
        shift = np.array([0.5, 0.5, 0.5]) - center_scaled_pos
        new_scaled_positions = scaled_positions + shift

        pbc = work_atoms.pbc
        for i in range(3):
            if pbc[i]:
                new_scaled_positions[:, i] = new_scaled_positions[:, i] % 1.0

        work_atoms.set_scaled_positions(new_scaled_positions)

        # 2. Geometric Cutout
        positions = work_atoms.get_positions()
        center_cart_pos = positions[self.center_index]

        lower_bound = center_cart_pos - self.box_vector / 2.0
        upper_bound = center_cart_pos + self.box_vector / 2.0

        mask = np.all((positions >= lower_bound) & (positions <= upper_bound), axis=1)
        initial_indices = set(np.where(mask)[0])

        # 3. Connectivity Expansion (Heal)
        if skin_depth > 0:
            # Build neighbor list for the FULL atoms
            # Multiplier 1.2 as requested
            cutoffs = natural_cutoffs(work_atoms, mult=1.2)
            nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
            nl.update(work_atoms)

            current_indices = initial_indices.copy()

            for _ in range(skin_depth):
                new_neighbors = set()
                for idx in current_indices:
                    indices, offsets = nl.get_neighbors(idx)
                    for neighbor_idx in indices:
                        if neighbor_idx not in current_indices:
                            new_neighbors.add(neighbor_idx)
                current_indices.update(new_neighbors)

            final_indices = list(current_indices)
        else:
            final_indices = list(initial_indices)

        if len(final_indices) <= 1:
             raise ValueError(f"Carved box contains too few atoms ({len(final_indices)}).")

        # Sort indices to preserve order (not strictly required but good for determinism)
        final_indices.sort()
        carved_atoms = work_atoms[final_indices]

        # Identify skin atoms (those NOT in initial geometric cut)
        # Note: indices in carved_atoms are 0..N. We need to map back or rely on positions.
        # But we know which ones were added.
        # Let's verify stoichiometry
        self._check_stoichiometry(carved_atoms)

        # 4. Re-wrapping
        if cluster_mode:
            carved_atoms.pbc = False
            carved_atoms.center(vacuum=10.0)
        else:
            # Periodic Mode logic
            new_center = self.box_vector / 2.0
            offset = new_center - center_cart_pos
            carved_atoms.positions += offset
            carved_atoms.set_cell(self.box_vector)
            carved_atoms.pbc = True

        # 5. Pre-relaxation
        if calculator is not None:
            self._pre_relax(carved_atoms, calculator, initial_indices, final_indices)

        return carved_atoms

    def _check_stoichiometry(self, cluster: Atoms):
        """Check if cluster stoichiometry deviates from bulk."""
        bulk_syms = self.atoms.get_chemical_symbols()
        cluster_syms = cluster.get_chemical_symbols()

        from collections import Counter
        bulk_counts = Counter(bulk_syms)
        cluster_counts = Counter(cluster_syms)

        # Simple ratio check
        bulk_total = len(bulk_syms)
        cluster_total = len(cluster_syms)

        for elem, count in bulk_counts.items():
            bulk_ratio = count / bulk_total
            cluster_ratio = cluster_counts.get(elem, 0) / cluster_total

            if abs(bulk_ratio - cluster_ratio) > 0.15: # 15% tolerance
                logger.warning(f"Stoichiometry mismatch for {elem}: Bulk={bulk_ratio:.2f}, Cluster={cluster_ratio:.2f}")

    def _pre_relax(self, cluster: Atoms, calculator: Calculator, initial_indices_set, final_indices_list):
        """
        Relax the cluster with fixed boundary.

        Boundary atoms are those that were NOT in the initial geometric cut.
        """
        cluster.calc = calculator

        # Map original indices to new indices
        # final_indices_list contains the indices in work_atoms that correspond to cluster[0], cluster[1]...
        # We need to find which of these were NOT in initial_indices_set.

        fixed_indices = []
        for i, original_idx in enumerate(final_indices_list):
            if original_idx not in initial_indices_set:
                fixed_indices.append(i)

        if fixed_indices:
            c = FixAtoms(indices=fixed_indices)
            cluster.set_constraint(c)

        # Run optimization
        # Use a temp file for log or None
        with tempfile.NamedTemporaryFile(mode='w', delete=True) as tmp:
            try:
                # 5-10 steps
                dyn = BFGS(cluster, logfile=tmp.name)
                dyn.run(fmax=0.5, steps=10) # Relax coarse
            except Exception as e:
                logger.warning(f"Pre-relaxation failed: {e}")

        # Remove constraint
        cluster.set_constraint()
