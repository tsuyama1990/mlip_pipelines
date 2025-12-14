
import numpy as np
from ase import Atoms
from lammps import lammps
import ctypes

from src.core.exceptions import UncertaintyInterrupt

class LammpsMaceDriver:
    def __init__(self, potential, threshold=0.1):
        self.potential = potential
        self.threshold = threshold
        # Initialize LAMMPS instance
        self.lmp = lammps(cmdargs=["-screen", "none", "-log", "none"])

    def _callback(self, caller, ntimestep, nlocal, tag, x, f):
        """
        Callback function for LAMMPS to get forces from MACE.

        Parameters:
            tag: Pointer to atom tags (IDs) (integers).
            x: Pointer to atom coordinates (doubles).
            f: Pointer to atom forces (doubles).
        """
        if not hasattr(self, 'current_atoms'):
             return

        # 1. Convert LAMMPS data to numpy
        coords = np.ctypeslib.as_array(x, shape=(nlocal, 3))

        # Handle sorting via tags
        # Tags in LAMMPS are 1-based IDs.
        # We assume they correspond to the 1-based indices of the original ASE atoms.
        # If tag is None (should not happen in real LAMMPS callback), we assume identity.

        if tag:
            tags = np.ctypeslib.as_array(tag, shape=(nlocal,))
            # argsort gives indices that sort the array: tags[indices[0]] is the smallest tag.
            # Assuming tags are 1..N, sorting them gives us the permutation to match ASE order (0..N-1).
            sort_indices = np.argsort(tags)

            # Reorder coords to match ASE atoms order
            sorted_coords = coords[sort_indices]
        else:
            # Fallback/Test support
            sort_indices = np.arange(nlocal)
            sorted_coords = coords

        if len(self.current_atoms) != nlocal:
             # Parallel execution or mismatch - Not fully supported in MVP
             pass

        self.current_atoms.set_positions(sorted_coords)

        # 2. Predict forces
        # potential.predict returns (energy, forces, stress)
        # Forces are in ASE order (sorted by ID)
        _, forces, _ = self.potential.predict(self.current_atoms)

        # 3. Check Uncertainty
        uncertainty = self.potential.get_uncertainty(self.current_atoms)
        if np.any(uncertainty > self.threshold):
            raise UncertaintyInterrupt(self.current_atoms.copy(), uncertainty)

        # 4. Write forces back to LAMMPS
        # We need to map forces (which are in ASE order) back to LAMMPS order.
        # f_array[i] should receive the force for the atom at LAMMPS index i.
        # sort_indices[k] = i  means the k-th atom (in sorted order) is at LAMMPS index i.
        # So f_array[sort_indices[k]] = forces[k]

        f_array = np.ctypeslib.as_array(f, shape=(nlocal, 3))
        f_array[sort_indices] = forces

    def run_md(self, atoms, lammps_script, threshold=None):
        """
        Run MD using the provided atoms and LAMMPS script.
        """
        if threshold is not None:
            self.threshold = threshold

        self.current_atoms = atoms.copy()

        # Reset LAMMPS
        self.lmp.command("clear")

        # Setup box
        self.lmp.command("units metal")
        self.lmp.command("atom_style atomic")
        self.lmp.command("boundary p p p")

        cell = atoms.get_cell()
        is_triclinic = not np.allclose(np.diag(np.diag(cell)), cell)

        xlo, ylo, zlo = 0.0, 0.0, 0.0
        xhi = cell[0,0]
        yhi = cell[1,1]
        zhi = cell[2,2]

        if is_triclinic:
             self.lmp.command(f"region mybox prism {xlo} {xhi} {ylo} {yhi} {zlo} {zhi} {cell[1,0]} {cell[2,0]} {cell[2,1]}")
        else:
             self.lmp.command(f"region mybox block {xlo} {xhi} {ylo} {yhi} {zlo} {zhi}")

        # Count types
        atomic_numbers = atoms.get_atomic_numbers()
        unique_types = sorted(list(set(atomic_numbers)))
        type_map = {z: i+1 for i, z in enumerate(unique_types)}
        num_types = len(unique_types)

        self.lmp.command(f"create_box {num_types} mybox")

        # Create atoms
        for i, atom in enumerate(atoms):
            itype = type_map[atom.number]
            pos = atom.position
            self.lmp.command(f"create_atoms {itype} single {pos[0]} {pos[1]} {pos[2]}")

        # Set masses (required)
        for z in unique_types:
            mass = next(a.mass for a in atoms if a.number == z)
            itype = type_map[z]
            self.lmp.command(f"mass {itype} {mass}")

        # Inject Bridge Code
        self.lmp.command("pair_style none")
        self.lmp.command("fix MACE_FORCE all external pf/callback 1 1")
        self.lmp.set_fix_external_callback("MACE_FORCE", self._callback, self)

        # Run user script
        try:
            for line in lammps_script.strip().split('\n'):
                self.lmp.command(line.strip())

            return self.current_atoms, "FINISHED"

        except UncertaintyInterrupt as e:
            return e.atoms, "UNCERTAIN"
        except Exception as e:
            raise e
