import logging
import typing

import numpy as np
from ase import Atom, Atoms
from ase.constraints import FixAtoms
from ase.neighborlist import neighbor_list
from ase.optimize import LBFGS

from src.domain_models.config import CutoutConfig
from src.domain_models.dtos import CutoutResult

logger = logging.getLogger(__name__)


def _extract_spherical_zones(
    atoms: Atoms, center_idx: int, core_r: float, buffer_r: float
) -> Atoms:
    # Use neighbor_list to find all atoms within buffer_r of the center_idx
    # The valid quantities are i, j, d, D, S, etc. Space separated usually or without commas.
    # ase's primitive_neighbor_list expects string like 'ijdD'
    i_arr, j_arr, d_arr, D_arr = neighbor_list("ijdD", atoms, buffer_r)

    mask = i_arr == center_idx
    neighbors_j = j_arr[mask]
    distances = d_arr[mask]
    vectors = D_arr[mask]

    # Include the center atom itself
    cluster_indices: np.ndarray = np.append(neighbors_j, center_idx)
    cluster_distances: np.ndarray = np.append(distances, 0.0)
    cluster_vectors = np.vstack([vectors, np.zeros(3)])

    # Sort by distance
    sort_idx = np.argsort(cluster_distances)
    cluster_indices = cluster_indices[sort_idx]
    cluster_distances = cluster_distances[sort_idx]
    cluster_vectors = cluster_vectors[sort_idx]

    # Create the cluster atoms object
    symbols = np.array(atoms.get_chemical_symbols())[cluster_indices]

    # positions are the vectors from the center atom
    cluster = Atoms(symbols=symbols, positions=cluster_vectors)
    cluster.set_pbc([False, False, False])

    # Assign force weights
    force_weights = np.zeros(len(cluster))
    core_mask = cluster_distances <= core_r
    force_weights[core_mask] = 1.0

    cluster.arrays["force_weights"] = force_weights

    return cluster


def _calculate_coordinations(cluster: Atoms) -> np.ndarray:
    from ase.neighborlist import NeighborList, natural_cutoffs

    cutoffs = natural_cutoffs(cluster)
    cutoffs = [c * 1.2 for c in cutoffs]
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
                dir_outward = vec_outward / norm if norm > 1e-5 else np.array([1.0, 0.0, 0.0])

                new_pos = cluster.positions[i] + dir_outward * 1.0
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


def _pre_relax_buffer(cluster: Atoms, mock_calculator: typing.Any) -> None:
    if mock_calculator is None:
        return

    cluster.calc = mock_calculator

    force_weights = cluster.arrays["force_weights"]
    core_indices = np.where(force_weights == 1.0)[0]

    constraint = FixAtoms(indices=core_indices)
    cluster.set_constraint(constraint)

    import contextlib

    try:
        opt = LBFGS(cluster, logfile=None)
        with contextlib.suppress(Exception):
            opt.run(fmax=0.1, steps=50)
    finally:
        cluster.calc = None
        cluster.set_constraint()


def extract_intelligent_cluster(
    atoms: Atoms, center_idx: int, cutout_config: CutoutConfig, mock_calculator: typing.Any = None
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

    if cutout_config.enable_pre_relaxation and mock_calculator is not None:
        _pre_relax_buffer(cluster, mock_calculator)

    return CutoutResult(cluster=cluster, passivation_atoms_added=passivation_count)
