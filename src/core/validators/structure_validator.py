import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from core.exceptions import InvalidStructureError
from core.constants import MIN_DISTANCE_RATIO

def validate_no_atomic_clash(atoms: Atoms, min_distance_ratio: float = MIN_DISTANCE_RATIO) -> None:
    """
    Verify no atoms are closer than min_distance_ratio * sum of covalent radii.

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object.
    min_distance_ratio : float, optional
        Minimum allowed distance as fraction of covalent radii sum, by default MIN_DISTANCE_RATIO.

    Raises
    ------
    InvalidStructureError
        If atomic clash detected or coordinates contain NaN/Inf.
    """
    if len(atoms) <= 1:
        return

    # Check for NaN/Inf coordinates first
    if np.any(np.isnan(atoms.positions)) or np.any(np.isinf(atoms.positions)):
        raise InvalidStructureError("Structure contains NaN or Inf coordinates")

    distances = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)

    # We iterate to find specific violations, but we can also do it vectorised for speed first
    # However, to give detailed error message, iteration is okay if N is small.
    # For performance on larger systems, we could optimize this.

    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            r_ij = distances[i, j]
            r_cov_sum = covalent_radii[atoms.numbers[i]] + covalent_radii[atoms.numbers[j]]
            min_allowed = r_cov_sum * min_distance_ratio

            if r_ij < min_allowed:
                raise InvalidStructureError(
                    f"Atomic clash detected: atoms {i}-{j} at {r_ij:.3f} Å "
                    f"(minimum allowed: {min_allowed:.3f} Å)"
                )

def validate_cell(atoms: Atoms) -> None:
    """
    Ensure cell matrix is non-singular and valid.

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object.

    Raises
    ------
    InvalidStructureError
        If cell volume is degenerate or cell contains NaN/Inf values.
    """
    if not np.any(atoms.pbc):
        return # Non-periodic systems don't strictly need a valid cell volume in same way

    # Check for NaN/Inf in cell
    if np.any(np.isnan(atoms.cell)) or np.any(np.isinf(atoms.cell)):
        raise InvalidStructureError("Cell contains NaN or Inf values")

    volume = atoms.get_volume()
    if volume < 1e-6:
        raise InvalidStructureError(f"Degenerate cell: volume = {volume:.2e} Å³")

def validate_structure(atoms: Atoms) -> None:
    """
    Run all structure validations.

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object.

    Raises
    ------
    InvalidStructureError
        If any validation fails or structure is empty.
    """
    if len(atoms) == 0:
        raise InvalidStructureError("Empty structure (0 atoms)")

    validate_cell(atoms)
    validate_no_atomic_clash(atoms)
