import tempfile
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.calculators.singlepoint import SinglePointCalculator
from loguru import logger

from core.exceptions import OracleComputationError
from core.interfaces import AbstractOracle
from utils.sssp import load_sssp_db, validate_pseudopotentials


class QeOracle(AbstractOracle):
    def __init__(
        self,
        dft_command: str = "pw.x",
        pseudo_dir: Path = Path("data/sssp"),
        sssp_json_path: Path = Path("data/sssp/SSSP_1.3.0_PBE_precision.json"),
        kpts_density: float = 0.15,  # 1/A
        nspin: int = 1
    ):
        """
        Initialize the Quantum ESPRESSO Oracle.

        Parameters
        ----------
        dft_command : str
            Command to run QE, e.g., "pw.x" or "mpirun -np 4 pw.x".
        pseudo_dir : Path
            Directory containing pseudopotential files.
        sssp_json_path : Path
            Path to the SSSP JSON metadata file.
        kpts_density : float
            K-point density in reciprocal Angstroms (1/A).
        nspin : int
            Default spin polarization (used if heuristic fails or for overrides).
        """
        self.dft_command = dft_command
        self.pseudo_dir = pseudo_dir
        self.sssp_json_path = sssp_json_path
        self.kpts_density = kpts_density
        self.nspin = nspin

        # Validate SSSP availability
        if not self.sssp_json_path.exists():
            logger.warning(f"SSSP JSON not found at {self.sssp_json_path}. Ensure scripts/download_sssp.py is run.")
        else:
            self.sssp_db = load_sssp_db(str(self.sssp_json_path))

    def _determine_kpts(self, atoms: Atoms) -> tuple[int, int, int]:
        """
        Determine k-points based on lattice vectors and density.
        """
        if not all(atoms.pbc):
            # Cluster / Molecule -> Gamma point only
            return (1, 1, 1)

        # Reciprocal lattice vectors b_i
        # b_i length = 2 * pi / |a_i| * ... (ASE handles get_reciprocal_cell)
        recip_cell = atoms.get_reciprocal_cell()
        b_norms = np.linalg.norm(recip_cell, axis=1) # Lengths of b1, b2, b3

        # N_i = max(1, ceil(density * |b_i|))
        # Note: ASE's reciprocal cell definition includes the 2*pi factor?
        # ASE get_reciprocal_cell() returns 2*pi*(A^-1)^T usually?
        # Let's verify standard ASE usage.
        # Actually a common heuristic is N * |real_lattice| ~ constant (e.g. 50 A)
        # OR k_density * |reciprocal_lattice_vector_length_without_2pi| ??
        # The prompt says: N_i = max(1, ceil(density * |b_i|)) where b_i is reciprocal lattice vector length.

        # Let's stick to the prompt's formula literally.
        kpts = []
        for b_len in b_norms:
            n = int(np.ceil(self.kpts_density * b_len))
            kpts.append(max(1, n))

        return tuple(kpts) # type: ignore

    def _get_pseudos_and_cutoffs(self, elements: list[str]) -> tuple[dict[str, str], float, float]:
        """
        Get pseudopotential filenames and determine max cutoffs.
        Returns: (pseudos_dict, ecutwfc, ecutrho)
        """
        unique_elements = sorted(list(set(elements)))
        validate_pseudopotentials(str(self.pseudo_dir), unique_elements, self.sssp_db)

        pseudos = {}
        max_ecutwfc = 0.0
        max_ecutrho = 0.0

        for elem in unique_elements:
            info = self.sssp_db[elem]
            filename = info.get("filename", f"{elem}.upf")
            pseudos[elem] = filename

            # SSSP JSON typically contains cutoff recommendations
            # We take the maximum recommended cutoff among all elements in the system
            # info keys might be "cutoff_wfc", "cutoff_rho" (in Ry)

            # Helper to safely get cutoff
            wfc = info.get("cutoff_wfc", 30.0) # Default 30 Ry if missing
            rho = info.get("cutoff_rho", wfc * 4) # Default 4x wfc

            if wfc > max_ecutwfc:
                max_ecutwfc = wfc
            if rho > max_ecutrho:
                max_ecutrho = rho

        return pseudos, max_ecutwfc, max_ecutrho

    def _determine_magnetization(self, atoms: Atoms) -> tuple[int, dict]:
        """
        Determine spin polarization and starting magnetization.
        Returns: (nspin, starting_magnetization_dict)
        """
        # Heuristic: Check for magnetic elements (3d transition metals + some others)
        magnetic_elements = {
            'Cr', 'Mn', 'Fe', 'Co', 'Ni', # 3d
            'Gd', 'Dy', 'Tb', 'Ho', 'Er', 'Tm' # Rare earths (basic list)
        }

        symbols = set(atoms.get_chemical_symbols())
        has_magnetic = not symbols.isdisjoint(magnetic_elements)

        if not has_magnetic:
            return 1, {}

        # Magnetic case
        # We need to map species index to magnetization, but ASE's Espresso calculator
        # usually handles dicts if we pass `magnetic_moments` or we can set it in system.
        # Actually, ASE maps `magmoms` or `initial_magmoms` to `starting_magnetization(i)`.
        # Here we will return a dict that we can merge into system block.
        # However, QE `starting_magnetization(i)` requires knowing the species index 'i'.
        # ASE handles this if we set `atoms.set_initial_magnetic_moments(...)`.

        # Strategy: We will set initial magnetic moments on the atoms object.
        # The calculator will pick it up.
        # But we also need to enforce nspin=2 in input_data.

        return 2, {}

    def compute(self, atoms: Atoms) -> Atoms:
        """
        Executes SCF calculation using ASE Espresso Calculator.
        """
        # 1. Setup Pseudopotentials & Cutoffs
        elements = atoms.get_chemical_symbols()
        try:
            pseudos, ecutwfc, ecutrho = self._get_pseudos_and_cutoffs(elements)
        except Exception as e:
            raise OracleComputationError(f"SSSP Setup Failed: {e}") from e

        # 2. Setup Calculator Profile
        # dft_command e.g. "mpirun -np 4 pw.x" -> argv=['mpirun', '-np', '4', 'pw.x']
        profile = EspressoProfile(argv=self.dft_command.split())

        # Determine Magnetism
        calc_nspin, _ = self._determine_magnetization(atoms)

        # If user forced nspin=2 in init, respect it?
        # Or let heuristic decide? Prompt says "heuristic ... is incorporated".
        # Let's use the max of init (default 1) and heuristic.
        final_nspin = max(self.nspin, calc_nspin)

        # 3. Execution Context
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir)

            # 4. Configure input_data
            input_data = {
                'control': {
                    'calculation': 'scf',
                    'restart_mode': 'from_scratch',
                    'disk_io': 'none',  # minimize IO
                    'tprnfor': True,    # Forces
                    'tstress': True,    # Stress
                    'pseudo_dir': str(self.pseudo_dir.resolve()),
                },
                'system': {
                    'ecutwfc': ecutwfc,
                    'ecutrho': ecutrho,
                    'occupations': 'smearing',
                    'smearing': 'mv',
                    'degauss': 0.01,
                    'nspin': final_nspin,
                },
                'electrons': {
                    'mixing_beta': 0.7,
                    'conv_thr': 1.0e-6
                }
            }

            # Apply starting magnetization if nspin=2
            # We set a default guess for all atoms if nspin=2 to help convergence/breaking symmetry
            calc_atoms = atoms.copy()
            if final_nspin == 2:
                # Set a simple default initial moment for all atoms (e.g. 0.5)
                # or just the magnetic ones.
                # Heuristic: Set 1.0 for everyone to allow symmetry breaking.
                # Or verify elements.
                mag_elements = {'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Gd', 'Dy', 'Tb', 'Ho', 'Er', 'Tm'}
                moms = []
                for s in calc_atoms.get_chemical_symbols():
                    if s in mag_elements:
                        moms.append(2.0) # Strong start
                    else:
                        moms.append(0.0)
                calc_atoms.set_initial_magnetic_moments(moms)

            # 5. Attach Calculator
            # Make a copy of atoms to not pollute the input instance with a transient calculator
            calc_atoms = atoms.copy()

            calc_atoms.calc = Espresso(
                profile=profile,
                input_data=input_data,
                pseudopotentials=pseudos,
                kpts=self._determine_kpts(calc_atoms),
                directory=tmpdir
            )

            try:
                # 6. Run Calculation
                # get_potential_energy triggers the calculation
                energy = calc_atoms.get_potential_energy()
                forces = calc_atoms.get_forces()
                stress = calc_atoms.get_stress()

                # 7. Detach & Clean
                # Create a result object with SinglePointCalculator
                result_atoms = atoms.copy()
                result_atoms.calc = SinglePointCalculator(
                    result_atoms,
                    energy=energy,
                    forces=forces,
                    stress=stress
                )

                return result_atoms

            except Exception as e:
                # 8. Error Handling
                # Ideally read the output file here if possible for better errors
                # But tmpdir vanishes after this block
                logger.warning(f"QE Calculation failed for {calc_atoms.get_chemical_formula()}: {e}")
                raise OracleComputationError(f"QE Failed: {e}") from e
