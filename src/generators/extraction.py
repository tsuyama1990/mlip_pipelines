import logging

import numpy as np
from ase import Atom, Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms
from ase.neighborlist import neighbor_list
from ase.optimize import LBFGS

from src.domain_models.config import CutoutConfig
from src.domain_models.dtos import CutoutResult

logger = logging.getLogger(__name__)

# Constants extracted to prevent hardcoding
PASSIVATION_BOND_LENGTH = 1.0
PRE_RELAX_FMAX = 0.1
PRE_RELAX_MAX_STEPS = 50
CUTOFF_MULTIPLIER = 1.2
MIN_VECTOR_NORM = 1e-5


def _extract_spherical_zones(
    atoms: Atoms, center_idx: int, core_r: float, buffer_r: float
) -> Atoms:
    if not (0 <= center_idx < len(atoms)):
        msg = f"center_idx {center_idx} is out of bounds for atoms of length {len(atoms)}."
        raise ValueError(msg)
    if core_r <= 0 or buffer_r <= 0:
        msg = "core_r and buffer_r must be strictly positive."
        raise ValueError(msg)
    if core_r >= buffer_r:
        msg = "core_r must be strictly smaller than buffer_r."
        raise ValueError(msg)
    if buffer_r > 50.0:
        msg = "buffer_r exceeds maximum allowed cutoff limit of 50.0 A."
        raise ValueError(msg)

    i_arr, j_arr, d_arr, D_arr = neighbor_list("ijdD", atoms, buffer_r)

    mask = i_arr == center_idx
    neighbors_j = j_arr[mask]
    distances = d_arr[mask]
    vectors = D_arr[mask]

    valid_mask = neighbors_j != center_idx
    neighbors_j = neighbors_j[valid_mask]
    distances = distances[valid_mask]
    vectors = vectors[valid_mask]

    cluster_indices: np.ndarray = np.append(neighbors_j, center_idx)

    if len(cluster_indices) > 5000:
        msg = f"Extraction resulted in an excessively large cluster ({len(cluster_indices)} atoms)."
        raise ValueError(msg)

    cluster_distances: np.ndarray = np.append(distances, 0.0)
    cluster_vectors = np.vstack([vectors, np.zeros(3)]) if len(vectors) > 0 else np.zeros((1, 3))

    sort_idx = np.argsort(cluster_distances)
    cluster_indices = cluster_indices[sort_idx]
    cluster_distances = cluster_distances[sort_idx]
    cluster_vectors = cluster_vectors[sort_idx]

    symbols = np.array(atoms.get_chemical_symbols())[cluster_indices]

    cluster = Atoms(symbols=symbols, positions=cluster_vectors)
    cluster.set_pbc([False, False, False])

    force_weights = np.zeros(len(cluster))
    core_mask = cluster_distances <= core_r
    force_weights[core_mask] = 1.0

    cluster.arrays["force_weights"] = force_weights

    return cluster


def _calculate_coordinations(cluster: Atoms) -> np.ndarray:
    from ase.neighborlist import NeighborList, natural_cutoffs

    cutoffs = natural_cutoffs(cluster)
    cutoffs = [c * CUTOFF_MULTIPLIER for c in cutoffs]
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(cluster)

    coordinations: np.ndarray = np.zeros(len(cluster), dtype=int)
    for i in range(len(cluster)):
        indices, _ = nl.get_neighbors(i)
        coordinations[i] = len(indices)
    return coordinations


def _get_expected_coordinations(
    symbols: list[str], coordinations: np.ndarray, force_weights: np.ndarray
) -> dict[str, int]:
    expected_coordination = {}
    for el in set(symbols):
        core_el_coords = [
            coordinations[i]
            for i in range(len(symbols))
            if symbols[i] == el and force_weights[i] == 1.0
        ]
        if core_el_coords:
            expected_coordination[el] = int(max(core_el_coords))
        else:
            el_coords = [coordinations[i] for i in range(len(symbols)) if symbols[i] == el]
            expected_coordination[el] = int(max(el_coords)) if el_coords else 0
    return expected_coordination


def _passivate_surface(cluster: Atoms, passivating_element: str = "H") -> int:
    """Passivates under-coordinated atoms on the surface of the cluster."""
    if len(cluster) == 0:
        return 0

    coordinations = _calculate_coordinations(cluster)
    symbols = cluster.get_chemical_symbols()
    force_weights = cluster.arrays["force_weights"]

    expected_coordination = _get_expected_coordinations(symbols, coordinations, force_weights)

    com = cluster.get_center_of_mass()
    new_atoms = []

    for i in range(len(cluster)):
        if force_weights[i] == 0.0:
            el = symbols[i]
            if coordinations[i] < expected_coordination.get(el, 0):
                vec_outward = cluster.positions[i] - com
                norm = float(np.linalg.norm(vec_outward))
                dir_outward = (
                    vec_outward / norm if norm > MIN_VECTOR_NORM else np.array([1.0, 0.0, 0.0])
                )

                new_pos = cluster.positions[i] + dir_outward * PASSIVATION_BOND_LENGTH
                new_atom = Atom(passivating_element, position=new_pos)
                new_atoms.append(new_atom)

    if not new_atoms:
        return 0

    for a in new_atoms:
        cluster.append(a)

    new_fw = np.zeros(len(cluster))
    new_fw[: len(force_weights)] = force_weights
    cluster.arrays["force_weights"] = new_fw

    return len(new_atoms)


def _pre_relax_buffer(cluster: Atoms, calculator: Calculator | None) -> None:
    if calculator is None:
        return

    cluster.calc = calculator

    force_weights = cluster.arrays["force_weights"]
    core_indices = np.where(force_weights == 1.0)[0]

    constraint = FixAtoms(indices=core_indices)
    cluster.set_constraint(constraint)

    import contextlib

    try:
        opt = LBFGS(cluster, logfile=None)
        with contextlib.suppress(Exception):
            opt.run(fmax=PRE_RELAX_FMAX, steps=PRE_RELAX_MAX_STEPS)
    finally:
        cluster.calc = None
        cluster.set_constraint()


def extract_intelligent_cluster(
    atoms: Atoms, center_idx: int, cutout_config: CutoutConfig, calculator: Calculator | None = None
) -> CutoutResult:
    """Intelligently extracts a spherical cluster from a bulk structure."""
    cluster = _extract_spherical_zones(
        atoms, center_idx, cutout_config.core_radius, cutout_config.buffer_radius
    )

    passivation_count = 0
    if cutout_config.enable_passivation:
        passivation_count = _passivate_surface(
            cluster, passivating_element=cutout_config.passivation_element
        )

    if cutout_config.enable_pre_relaxation and calculator is not None:
        _pre_relax_buffer(cluster, calculator)

    return CutoutResult(cluster=cluster, passivation_atoms_added=passivation_count)
