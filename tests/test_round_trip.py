import numpy as np
from ase.io import read, write
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator

class TestRoundTrip:
    def test_xyz_save_load(self, tmp_path):
        """MASD 5.2: XYZ round-trip preserves data"""
        atoms_orig = bulk('Si', cubic=True)
        # Use SinglePointCalculator to properly attach results
        energy = -10.5
        forces = np.random.rand(len(atoms_orig), 3)
        calc = SinglePointCalculator(atoms_orig, energy=energy, forces=forces)
        atoms_orig.calc = calc

        filepath = tmp_path / "test.xyz"
        write(filepath, atoms_orig, format='extxyz')
        atoms_loaded = read(filepath, format='extxyz')

        # When reading extxyz, energy/forces often land in calculator
        # ASE's extxyz reader usually attaches a SinglePointCalculator

        # Check positions
        assert np.allclose(atoms_orig.positions, atoms_loaded.positions, atol=1e-10)

        # Check energy
        # If calc exists, get_potential_energy() works
        assert atoms_loaded.calc is not None
        assert np.isclose(atoms_orig.get_potential_energy(), atoms_loaded.get_potential_energy())

        # Check forces
        assert np.allclose(atoms_orig.get_forces(), atoms_loaded.get_forces())
