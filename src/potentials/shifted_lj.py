import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import NeighborList

class ShiftedLennardJones(Calculator):
    """
    Shifted Lennard-Jones potential: V(r) = V_LJ(r) - V_LJ(rc) for r < rc, else 0.
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, sigma=1.0, epsilon=1.0, cutoff=2.5, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.epsilon = epsilon
        self.cutoff = cutoff
        # Value at cutoff
        sr6 = (sigma/cutoff)**6
        sr12 = sr6**2
        self.v_rc = 4 * epsilon * (sr12 - sr6)

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        # Helper to get neighbors
        # We use a neighbor list with cutoff
        # The cutoff for neighbor list should be self.cutoff.
        # NeighborList takes 'cutoffs' as list of radii.
        # If we pass list of radii, the effective cutoff is r_i + r_j.
        # So we pass cutoff/2 for all atoms.

        n_atoms = len(self.atoms)
        cutoffs = [self.cutoff / 2.0] * n_atoms
        nl = NeighborList(cutoffs, skin=0.0, self_interaction=False, bothways=True)
        nl.update(self.atoms)

        positions = self.atoms.positions
        cell = self.atoms.cell
        pbc = self.atoms.pbc

        energy = 0.0
        forces = np.zeros((n_atoms, 3))

        for i in range(n_atoms):
            indices, offsets = nl.get_neighbors(i)

            for j, offset in zip(indices, offsets):
                # Vector from i to j
                r_vec = positions[j] + np.dot(offset, cell) - positions[i]
                r2 = np.sum(r_vec**2)

                if r2 > self.cutoff**2 or r2 == 0:
                    continue

                r = np.sqrt(r2)

                # LJ terms
                sr2 = (self.sigma / r)**2
                sr6 = sr2**3
                sr12 = sr6**2

                # Energy: 4eps(sr12 - sr6) - V_rc
                v = 4 * self.epsilon * (sr12 - sr6) - self.v_rc
                energy += v

                # Force scalar: -dV/dr / r
                # dV/dr = 24*eps/r * (sr6 - 2*sr12)
                # F_vec = - (dV/dr) * (r_vec/r) = - (dV/dr)/r * r_vec
                #       = - 24*eps/r^2 * (sr6 - 2*sr12) * r_vec
                #       = 24*eps/r^2 * (2*sr12 - sr6) * r_vec

                f_prefactor = (24 * self.epsilon / r2) * (2 * sr12 - sr6)
                forces[i] += f_prefactor * r_vec

        self.results['energy'] = energy / 2.0
        self.results['forces'] = forces
