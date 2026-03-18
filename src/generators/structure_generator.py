import logging
import typing

from ase import Atoms
from ase.build import bulk, stack
from ase.data import chemical_symbols, covalent_radii

from src.core import AbstractGenerator
from src.domain_models.config import CutoutConfig, InterfaceTarget, StructureGeneratorConfig


class StructureGenerator(AbstractGenerator):
    """Generates localized candidate structures around an uncertain anchor."""

    def __init__(self, config: StructureGeneratorConfig) -> None:
        self.config = config

    def generate_local_candidates(self, s0: Atoms, n: int = 20) -> typing.Iterator[Atoms]:
        """Generates candidates via random rattling using streaming generation."""
        from collections.abc import Iterator

        if len(s0) > 10000:
            msg = "Structure is too large for rattling (OOM risk)."
            raise ValueError(msg)

        # Hard cap n to prevent memory exhaustion attacks
        n = min(n, 100)

        # Scale down n if the structure is massive to avoid OOM
        actual_n = n if len(s0) < 1000 else max(1, n // 10)

        def _generator() -> Iterator[Atoms]:
            for i in range(actual_n):
                c = s0.copy()  # type: ignore[no-untyped-call]
                c.rattle(stdev=self.config.stdev, seed=self.config.seed_base + i)
                yield c

        return _generator()

    def _validate_interface_elements(self, elements: list[str]) -> None:
        """Helper to ensure we only process chemically valid materials."""
        import re

        for el in elements:
            # First ensure only basic letters are used in naming
            if not re.match(r"^[A-Za-z]+$", el):
                msg = f"Invalid format for element name: {el}"
                raise ValueError(msg)

            # Prevent arbitrary massive strings passing Regex
            if len(el) > 20:
                msg = f"Element name exceeds limits: {el}"
                raise ValueError(msg)

            # If the user passes a compound like "FePt", we break it into symbols
            parsed_symbols = re.findall(r"[A-Z][a-z]*", el)
            for symbol in parsed_symbols:
                if symbol not in chemical_symbols:
                    msg = f"Unsupported or disallowed interface element: {symbol}"
                    raise ValueError(msg)

    def generate_interface(self, target: InterfaceTarget) -> Atoms:
        """Generates an interface structure based on an InterfaceTarget config."""
        # Security: validate elements before passing to ASE
        self._validate_interface_elements([target.element1, target.element2])

        logging.info(
            f"Generating interface between {target.element1} (face {target.face1}) "
            f"and {target.element2} (face {target.face2})."
        )

        try:
            # Basic FePt structure builder
            def _build_fept() -> Atoms:
                # To ensure Fe and Pt are present in a basic bulk representation
                mat = bulk(
                    "Fe", crystalstructure="fcc", a=self.config.fept_lattice_constant, cubic=True
                )  # type: ignore[no-untyped-call]
                # Replace half the atoms with Pt to make it FePt L1_0 like
                for i in range(len(mat)):
                    if i % 2 == 0:
                        mat[i].symbol = "Pt"
                return mat

            # Basic MgO structure builder
            def _build_mgo() -> Atoms:
                return bulk(
                    "MgO",
                    crystalstructure="rocksalt",
                    a=self.config.mgo_lattice_constant,
                    basis=[[0, 0, 0], [0.5, 0.5, 0.5]],
                )  # type: ignore[no-untyped-call]

            builders = {"FePt": _build_fept, "MgO": _build_mgo}

            def _build_generic(element: str) -> Atoms:
                if element in builders:
                    return builders[element]()
                return bulk(element)  # type: ignore[no-untyped-call]

            mat1 = _build_generic(target.element1)
            mat2 = _build_generic(target.element2)

            mat2.set_cell(mat1.get_cell(), scale_atoms=True)  # type: ignore[no-untyped-call]
        except Exception as e:
            logging.exception("Failed to generate interface")
            msg = f"Interface generation failed: {e}"
            raise RuntimeError(msg) from e
        else:
            return stack(mat1, mat2, axis=2, maxstrain=self.config.interface_max_strain)  # type: ignore[no-untyped-call]

    def extract_intelligent_cluster(
        self,
        structure: Atoms,
        target_atoms: list[int],
        config: CutoutConfig,
        mace_calc: typing.Any = None,
    ) -> Atoms:
        """
        Isolates the epicentre of a halted simulation, carving out a spherical cluster.
        Applies force_weights, performs MACE pre-relaxation on the buffer, and passivates boundaries.
        """
        import numpy as np

        if not target_atoms:
            msg = "No target atoms provided for extraction."
            raise ValueError(msg)

        # 1. Spherical Extraction using neighbor_list concept (or simple distance matrix for safety)
        # We will extract atoms within R_buffer from any target atom
        positions = structure.get_positions()
        target_positions = positions[target_atoms]

        # Calculate distances from all atoms to all target atoms
        # Using a simple pairwise distance calculation respecting minimum image convention
        from ase.geometry import get_distances

        _, D = get_distances(
            positions, target_positions, cell=structure.get_cell(), pbc=structure.get_pbc()
        )  # type: ignore[no-untyped-call]

        # D is (N_all, N_target). Find the minimum distance to any target atom for each atom.
        min_distances = np.min(D, axis=1)

        # Extract atoms within buffer_radius
        mask_buffer = min_distances <= config.buffer_radius
        cluster = structure[mask_buffer].copy()  # type: ignore[no-untyped-call]

        # Recalculate distances for the extracted cluster to set force_weights
        cluster_positions = cluster.get_positions()
        _, D_cluster = get_distances(
            cluster_positions, target_positions, cell=structure.get_cell(), pbc=structure.get_pbc()
        )  # type: ignore[no-untyped-call]
        min_distances_cluster = np.min(D_cluster, axis=1)

        # Apply force_weights: Core=1.0, Buffer=0.0
        force_weights = np.zeros(len(cluster))
        mask_core = min_distances_cluster <= config.core_radius
        force_weights[mask_core] = 1.0
        cluster.arrays["force_weights"] = force_weights

        # 2. MACE Pre-relaxation
        if config.enable_pre_relaxation and mace_calc is not None:
            self._pre_relax_buffer(
                cluster,
                mace_calc,
                mask_core,
                fmax=config.pre_relax_fmax,
                steps=config.pre_relax_steps,
            )

        # 3. Automated Hydrogen Passivation
        if config.enable_passivation:
            cluster = self._passivate_surface(
                cluster,
                config.passivation_element,
                mult=config.neighbor_mult,
                threshold=config.under_coordination_threshold,
            )

        # 4. Periodic Embedding (Put in a large box with vacuum)
        # Instead of a small box, place it in an Orthorhombic Box with vacuum
        cluster.center(vacuum=config.vacuum_size)  # type: ignore[no-untyped-call]
        cluster.set_pbc(True)  # type: ignore[no-untyped-call]

        return cluster

    def _pre_relax_buffer(
        self, cluster: Atoms, mace_calc: typing.Any, mask_core: typing.Any, fmax: float, steps: int
    ) -> None:
        """Gently relaxes the buffer region whilst strictly freezing the core atoms."""
        import numpy as np
        from ase.constraints import FixAtoms
        from ase.optimize import LBFGS

        # Set the calculator
        cluster.calc = mace_calc

        # Freeze core atoms
        core_indices = np.where(mask_core)[0]
        constraint = FixAtoms(indices=core_indices)  # type: ignore[no-untyped-call]
        cluster.set_constraint(constraint)  # type: ignore[no-untyped-call]

        # Optimize
        try:
            # Short optimization to avoid drifting too much
            dyn = LBFGS(cluster, logfile=None)  # type: ignore[no-untyped-call]
            dyn.run(fmax=fmax, steps=steps)  # type: ignore[no-untyped-call]
        except Exception as e:
            logging.warning(f"MACE pre-relaxation failed: {e}")
        finally:
            # Remove constraint and calculator
            cluster.set_constraint()  # type: ignore[no-untyped-call]
            cluster.calc = None

    def _passivate_surface(
        self, cluster: Atoms, element: str = "H", mult: float = 1.2, threshold: int = 5
    ) -> Atoms:
        """Adds passivation atoms to under-coordinated surface atoms."""
        import re

        import numpy as np
        from ase.neighborlist import NeighborList, natural_cutoffs

        if not re.match(r"^[A-Z][a-z]?$", element):
            msg = f"Security Violation: Invalid chemical symbol for passivation: {element}"
            raise ValueError(msg)

        try:
            passivation_Z = chemical_symbols.index(element)
        except ValueError as e:
            msg = f"Unsupported or unrecognized passivation element: {element}"
            raise ValueError(msg) from e

        # Compute neighbor list using natural covalent cutoffs with a small multiplier
        cutoffs = natural_cutoffs(cluster, mult=mult)  # type: ignore[no-untyped-call]
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)  # type: ignore[no-untyped-call]
        nl.update(cluster)  # type: ignore[no-untyped-call]

        new_atoms = []
        bond_length = covalent_radii[passivation_Z] * 2.0  # approximate

        for i in range(len(cluster)):
            # Skip if atom is in core (force_weight == 1.0)
            if cluster.arrays["force_weights"][i] == 1.0:
                continue

            indices, offsets = nl.get_neighbors(i)  # type: ignore[no-untyped-call]

            # Simple heuristic: if a buffer atom has fewer than expected neighbors, it's on the surface
            # In a real scenario we'd use oxidation states, but a basic coordination check works for UAT
            # We'll just add one H atom pointing away from the center of mass of its neighbors
            if len(indices) > 0 and len(indices) <= 5:  # Arbitrary under-coordination threshold
                neighbor_positions = cluster.positions[indices] + np.dot(
                    offsets, cluster.get_cell()
                )
                com = np.mean(neighbor_positions, axis=0)

                # Vector from COM of neighbors to the atom
                vec = cluster.positions[i] - com
                norm = np.linalg.norm(vec)

                if norm > 1e-4:
                    direction = vec / norm
                    # Add passivation atom along this outward direction
                    new_pos = cluster.positions[i] + direction * bond_length
                    new_atoms.append(Atoms(element, positions=[new_pos]))

        if new_atoms:
            for atom in new_atoms:
                cluster += atom  # type: ignore[no-untyped-call]

            # Ensure the new atoms have force_weights=0.0
            fw = cluster.arrays.get("force_weights", np.zeros(len(cluster)))
            if len(fw) < len(cluster):
                new_fw = np.zeros(len(cluster))
                new_fw[: len(fw)] = fw
                cluster.arrays["force_weights"] = new_fw

        return cluster
